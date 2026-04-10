import numpy as np
import pandas as pd

def _build_pseudo_true_labels(metadata, syllable_labels, flat_index_map, repeat_n):
    """Constructs true labels dynamically from metadata and syllables for plotting."""

    if metadata is None and syllable_labels is None:
        return None, None
    
    true_label_arrays = []
    true_label_names = []
    
    if metadata is not None and not metadata.empty:
        meta_cols = [c for c in metadata.columns if c not in ["run_id", "animal_id", "trial_id", "strain"]]
        if len(meta_cols) > 0:
            # Label encode categorical metadata columns using sorted order to match plot_2D mapping
            encoded_meta = metadata[meta_cols].apply(lambda x: pd.factorize(x, sort=True)[0])
            meta_features = meta_cols
            
            run_to_meta = {}
            for i, row in metadata.iterrows():
                run_id = row.get("run_id")
                if run_id is not None:
                    run_to_meta[run_id] = encoded_meta.iloc[i].values
                    
            true_label_names.extend(meta_features)
            
            meta_matrix = []
            empty_meta = np.zeros(len(meta_features))
            for run_id, token_idx in flat_index_map:
                meta_matrix.append(run_to_meta.get(run_id, empty_meta))
            true_label_arrays.append(np.array(meta_matrix))
            
    if syllable_labels is not None:
        true_label_names.append("Syllables")
        syll_matrix = []
        for run_id, token_idx in flat_index_map:
            if run_id in syllable_labels and token_idx < len(syllable_labels[run_id]):
                syll_id = int(syllable_labels[run_id][token_idx])
                syll_matrix.append([syll_id])
            else:
                syll_matrix.append([-1])  # -1 for missing
        true_label_arrays.append(np.array(syll_matrix))
            
    if len(true_label_arrays) == 0:
        return None, None
        
    combined = np.concatenate(true_label_arrays, axis=1)
    if repeat_n > 1:
        combined = np.repeat(combined, repeat_n, axis=0)
        
    return combined, true_label_names


def _resolve_metric_for_data_type(args, data_type, metric_name):
    """Resolve per-data-type metric values with sensible fallback.

    Priority:
    - {data_type}_{metric_name}
    - raw_{metric_name}
    - {metric_name}
    """
    if data_type not in {"raw", "umap", "tsne"}:
        return getattr(args, metric_name)

    scoped_attr = f"{data_type}_{metric_name}"
    raw_attr = f"raw_{metric_name}"

    scoped_val = getattr(args, scoped_attr, None)
    if scoped_val is not None:
        return scoped_val

    raw_val = getattr(args, raw_attr, None)
    if raw_val is not None:
        return raw_val

    return getattr(args, metric_name)


def build_clustroid_full_map(clustroids_idxs, layer_flat_index_map, embeddings, kmeans_labels_per_layer, hdbscan_labels_per_layer, gmm_labels_per_layer):
    clustroids_index_map = {}
    clustroids_full_map = {}
    for layer_key, alg_dict in clustroids_idxs.items():
        clustroids_index_map[layer_key] = {}
        clustroids_full_map[layer_key] = {}
        for alg_name, cluster_indices in alg_dict.items():
            if len(cluster_indices) >= 50:
                print(f"Warning: {len(cluster_indices)} clustroids found for {layer_key} and {alg_name}, which may be too many for downstream analysis. Consider reducing the number of clusters or clustroids per cluster.")
                cluster_indices = cluster_indices[:50]
            clustroids_index_map[layer_key][alg_name] = [layer_flat_index_map[layer_key][idx] for idx in cluster_indices]

            if "kmeans" in alg_name:
                labels = kmeans_labels_per_layer[list(embeddings.keys()).index(layer_key)]
            elif "hdbscan" in alg_name:
                labels = hdbscan_labels_per_layer[list(embeddings.keys()).index(layer_key)]
            elif "gmm" in alg_name:
                labels = gmm_labels_per_layer[list(embeddings.keys()).index(layer_key)]
            else:
                labels = None
            clusters = []
            for i, clustroid_idx in enumerate(cluster_indices):
                if labels is not None:
                    cluster_id = labels[clustroid_idx]
                    member_indices = [layer_flat_index_map[layer_key][j] for j, lab in enumerate(labels) if lab == cluster_id]
                else:
                    member_indices = []
                clusters.append({
                    "clustroid": layer_flat_index_map[layer_key][clustroid_idx],
                    "members": member_indices
                })
            clustroids_full_map[layer_key][alg_name] = clusters
    return clustroids_full_map

def _build_alg_params(args, data_type, layer_count):
    """Returns {alg_name: [(params, subdir, label_fname), ...]} per layer."""
    kmeans_ks            = _resolve_metric_for_data_type(args, data_type, "kmeans_ks")
    hdbscan_cluster_sizes = _resolve_metric_for_data_type(args, data_type, "hdbscan_cluster_size")
    gmm_ks               = _resolve_metric_for_data_type(args, data_type, "gmm_ks")

    def _kmeans_entry(layer_idx):
        k = kmeans_ks[layer_idx]
        return ({"k": k, "random_state": args.seed},
                f"kmeans_k_{k}_analysis", f"kmeans_labels_k_{k}.npy")

    def _hdbscan_entry(layer_idx):
        cs = hdbscan_cluster_sizes[layer_idx]
        return ({"cluster_size": cs},
                f"hdbscan_cluster_size_{cs}_analysis", f"hdbscan_labels_cluster_size_{cs}.npy")

    def _gmm_entry(layer_idx):
        gmm_k = int(gmm_ks[layer_idx])
        return ({"k": gmm_k, "max_iter": args.gmm_max_iter, "tol": args.gmm_tol,
                 "reg_covar": args.gmm_reg_covar,
                 "random_state": getattr(args, "gmm_random_state", args.seed)},
                f"gmm_k_{gmm_k}_analysis", f"gmm_labels_k_{gmm_k}.npy")

    sources = {
        "kmeans":  (kmeans_ks,             _kmeans_entry),
        "hdbscan": (hdbscan_cluster_sizes,  _hdbscan_entry),
        "gmm":     (gmm_ks,                _gmm_entry),
    }
    return {
        alg: [builder(i) for i in range(min(layer_count, len(param_list)))]
        for alg, (param_list, builder) in sources.items()
    }

def _resolve_labels(true_labels, true_label_names, metadata, syllable_labels, embeddings, layer_flat_index_map, token_shapes, args):
    plot_true_labels = true_labels
    plot_true_label_names = true_label_names
    
    if true_labels is None and (metadata is not None or syllable_labels is not None):
        first_layer_key = list(embeddings.keys())[0]
        flat_map = layer_flat_index_map[first_layer_key]
        first_repeat_n = token_shapes[0][0] if token_shapes[0][0] > 0 else getattr(args, "base_window_size", 2)
        
        # Build pseudo-true labels specifically formatted for how plot_cluster_temporal expects array inputs
        pseudo_labels, pseudo_names = _build_pseudo_true_labels(
            metadata, syllable_labels, flat_map, first_repeat_n
        )
        if pseudo_labels is not None:
            # Wrap in dictionary to simulate existing true_label format 
            plot_true_labels = {"pseudo_run": pseudo_labels}
            plot_true_label_names = pseudo_names

    return plot_true_labels, plot_true_label_names
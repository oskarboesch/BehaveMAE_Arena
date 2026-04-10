import numpy as np
from cuml.cluster import HDBSCAN, KMeans
from cuml.cluster.hdbscan import approximate_predict


from .fit_gmm import _fit_gmm_pyro, _predict_gmm_pyro, _fit_gmm_sklearn_fallback, _sanitize_embeddings_for_gmm
from .cluster_utils import build_clustroid_full_map, _build_alg_params, _resolve_labels
from analysis.utils.window_and_aggregate import window_and_aggregate
from ..plot.plot_k_means_silhouette import plot_k_means_silhouettes
from ..plot.plot_hdbscan_silhouette import plot_hdbscan_silhouettes
from ..plot.plot_gmm_silhouette import plot_gmm_silhouettes
from ..plot.plot_2D import plot_2D, _cluster_color_map
from ..plot.plot_cluster_temporal import plot_cluster_temporal
from ..utils.get_clustroid import get_clustroid_idx
from ..utils.title_print import title_print
from .cluster_label_analysis import analyze_cluster_labels
from tqdm import tqdm
import os
import pandas as pd
import argparse
MAX_WINDOWS_FOR_FIT = 10 * 3200

def preprocess_for_cluster(embeddings: dict, args: argparse.Namespace):
    """Preprocess embeddings for clustering: windowing, dimensionality reduction, and mapping construction.
        Args:        
            embeddings: dict of {layer_key: {run_id: np.array(num_tokens, embedding_dim)}}
            args: argparse.Namespace with necessary parameters for windowing and dimensionality reduction.
        Returns:     
            list_of_embeddings: list of np.arrays for each layer, concatenated across runs, ready for clustering.
            list_of_embeddings_fit: list of np.arrays for each layer, used for fitting the clustering model.
            layer_flat_index_map: dict mapping layer_key to list of (run_id, token_idx_in_run) for each row in the corresponding list_of_embeddings entry.
    """
    list_of_embeddings = []
    list_of_embeddings_fit = []
    layer_flat_index_map = {}

    for layer_key, embeddings_dict in tqdm(embeddings.items(), desc="Preparing embeddings for clustering"):
        # window to simply select correct number of windows
        embedding_windowed, layer_window_map = window_and_aggregate(embeddings_dict, window_size=1, 
                                                    stride=1, method=args.agg_method, n_windows_per_run=args.n_windows_per_run_for_clustering, seed=args.seed)
        X = np.concatenate([v[:, :args.ndim_for_cluster] for v in embedding_windowed.values()], axis=0)

        if len(X) > MAX_WINDOWS_FOR_FIT:
            n_windows_fit = min(len(emb) for emb in embedding_windowed.values()) 
            embedding_fit, _ = window_and_aggregate(embeddings_dict, window_size=1, 
                                                        stride=1, method=args.agg_method, n_windows_per_run=min(10, n_windows_fit), seed=args.seed)
        else :
            embedding_fit = embedding_windowed
            
        # flatten the embeddings across all runs 
        print(f"Using {args.ndim_for_cluster} dimensions for clustering.")
        X_fit = np.concatenate([v[:, :args.ndim_for_cluster] for v in embedding_fit.values()], axis=0)
        print(f"Subsampled from {len(X)} to {len(X_fit)} windows for silhouette analysis and fitting.")


        # Explicit mapping from flattened row index -> (run_id, token_idx_in_run).
        flat_index_map = []
        for run_id, token_indices in layer_window_map.items():
            for token_idx in token_indices:
                flat_index_map.append((run_id, int(token_idx)))
        layer_flat_index_map[layer_key] = flat_index_map

        list_of_embeddings.append(X)
        list_of_embeddings_fit.append(X_fit)
        print(f"Prepared embeddings for {layer_key}: full shape {X.shape}, silhouette sample shape {X_fit.shape}")
    return list_of_embeddings, list_of_embeddings_fit, layer_flat_index_map

def kmeans_cluster(X: np.ndarray, X_fit: np.ndarray, k: int, random_state: int):
    if k >= X_fit.shape[0]:
        raise ValueError(f"k={k} must be less than the number of samples ({X_fit.shape[0]}) for KMeans.")
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X_fit)
    return kmeans.predict(X)

def hdbscan_cluster(X: np.ndarray, X_fit: np.ndarray, cluster_size: int):
    hdbscan = HDBSCAN(min_cluster_size=cluster_size, min_samples=cluster_size, prediction_data=True)
    hdbscan.fit(X_fit)
    return approximate_predict(hdbscan, X)[0]

def gmm_cluster(X: np.ndarray, X_fit: np.ndarray, k: int, max_iter: int, tol: float, reg_covar: float, random_state: int):
    if k >= X_fit.shape[0]:
        raise ValueError(f"k={k} must be less than the number of samples ({X_fit.shape[0]}) for GMM.")
    gmm_input = _sanitize_embeddings_for_gmm(X_fit)
    try:
        _, gmm_params = _fit_gmm_pyro(
            gmm_input,
            n_components=k,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        return _predict_gmm_pyro(X, gmm_params)
    except Exception as e:
        print(f"Warning: Pyro GMM failed: {e}. Falling back to sklearn GaussianMixture.")
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=k, reg_covar=reg_covar, random_state=random_state, max_iter=max_iter)
        gmm.fit(gmm_input)
        return gmm.predict(X)
    
def plot_silhouette_scores(X_fit: np.ndarray, args: argparse.Namespace, algorithm: str, output_path: str):
    if algorithm == "kmeans":
        plot_k_means_silhouettes(X_fit, k_range=args.kmeans_k_range, random_state=args.seed, savepath=output_path)
    elif algorithm == "hdbscan":
        plot_hdbscan_silhouettes(X_fit, cluster_size_range=args.hdbscan_cluster_size_range, savepath=output_path)
    elif algorithm == "gmm":
        plot_gmm_silhouettes(X_fit, k_range=args.gmm_ks_range, max_iter=args.gmm_max_iter, tol=args.gmm_tol, reg_covar=args.gmm_reg_covar, random_state=args.seed, savepath=output_path)
    else:
        raise ValueError(f"Unsupported algorithm for silhouette plotting: {algorithm}")
    
def cluster(X: np.ndarray, X_fit: np.ndarray, algorithm: str, alg_params: dict, 
            analysis: bool = False, flat_index_map: list = None, metadata: dict = None, 
            kinematics: dict = None, time_window_size: int = 10, output_dir: str = None):
    """Run specified clustering algorithm and optionally analyze cluster labels. Analysis includes mapping cluster labels back to metadata and kinematics, and saving correlation results as well as 2D cluster plots.
        Args:
            X: np.ndarray of shape (num_samples, embedding_dim) - the full set of embeddings to cluster.
            X_fit: np.ndarray of shape (num_fit_samples, embedding_dim) - the subset of embeddings used for fitting the clustering model (e.g., for silhouette analysis).
            algorithm: str - the clustering algorithm to use ("kmeans", "hdbscan", or "gmm").
            alg_params: dict - parameters specific to the chosen algorithm (e.g., {"k": 5} for kmeans).
            analysis: bool - whether to perform cluster label analysis and plotting.
            flat_index_map: list of (run_id, token_idx_in_run) tuples mapping each row of X back to original run and token indices, required for analysis.
            metadata: dict - mapping run_id to metadata dict, used for analysis.
            kinematics: dict - mapping run_id to kinematics dict, used for analysis.
            time_window_size: int - the size of the time window to consider when analyzing cluster labels against metadata and kinematics.
            output_dir: str - directory to save analysis results and plots, required if analysis is True.
        Returns:
            labels: np.ndarray of shape (num_samples,) containing cluster labels for each sample in X.
            clustroid_indices: list of indices corresponding to the clustroid of each cluster, useful for further analysis or visualization.
    """
    if algorithm == "kmeans":
        labels = kmeans_cluster(X, X_fit, **alg_params)
    elif algorithm == "hdbscan":
        labels = hdbscan_cluster(X, X_fit, **alg_params)
    elif algorithm == "gmm":
        labels = gmm_cluster(X, X_fit, **alg_params)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    clustroid_indices = []
    if analysis and flat_index_map is not None and output_dir is not None:
        analyze_cluster_labels(
            cluster_labels=labels.tolist(), 
            flat_index_map=flat_index_map, 
            metadata=metadata, 
            kinematics=kinematics, 
            time_window_size=time_window_size, 
            output_dir=output_dir
        )
        plot_2D(
            X,
            labels,
            output_path=os.path.join(output_dir, f"{algorithm}_cluster_plot.png"),
            title=f"{algorithm} Clustering",
        )
        clean_labels = labels[labels != -1] if algorithm == "hdbscan" else labels
        for cluster_id in np.unique(clean_labels):
            if np.any(clean_labels == cluster_id):
                clustroid_indices.append(get_clustroid_idx(X, clean_labels, cluster_id))
            else:
                print(f"Warning: Cluster {cluster_id} empty for {algorithm}.")
    return labels, clustroid_indices
    
def cluster_all_layers(
    alg_name: str, alg_params_per_layer: list, embeddings: dict, list_of_embeddings: list, list_of_embeddings_fit: list,
    layer_flat_index_map: dict, token_shapes: list, clustroids_idxs: dict, 
    output_dir: str, args: argparse.Namespace, metadata: pd.DataFrame=None, kinematics: pd.DataFrame=None,
):
    """Run one clustering algorithm across all layers, returning accumulators."""
    plot_labels, labels_per_layer, color_maps = [], [], []

    for layer_idx, layer_key in enumerate(embeddings.keys()):
        if layer_idx >= len(alg_params_per_layer):
            continue

        params, subdir, label_fname = alg_params_per_layer[layer_idx]
        repeat_n = token_shapes[layer_idx][0] if token_shapes[layer_idx][0] > 0 else getattr(args, "base_window_size", 2)
        output_path = os.path.join(output_dir, layer_key)
        fig_output_path = os.path.join(output_dir, "figures", layer_key)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(fig_output_path, exist_ok=True)
        clustroids_idxs.setdefault(layer_key, {})

        try:
            labels, clustroid_indices = cluster(
                X=list_of_embeddings[layer_idx], X_fit=list_of_embeddings_fit[layer_idx],
                algorithm=alg_name, alg_params=params, analysis=True,
                flat_index_map=layer_flat_index_map[layer_key], metadata=metadata,
                kinematics=kinematics, time_window_size=max(1, 750 // repeat_n),
                output_dir=os.path.join(fig_output_path, subdir)
            )
            clustroids_idxs[layer_key][subdir] = np.array(clustroid_indices).flatten()
            color_maps.append(_cluster_color_map(labels))
            print(f"Creating plot labels for {layer_key} with repeat_n={repeat_n}... and labels shape {labels.shape}")
            plot_labels.append(np.repeat(labels, repeat_n, axis=0))
            labels_per_layer.append(labels)
            np.save(os.path.join(output_path, label_fname), labels)
        except Exception as e:
            print(f"Warning: {alg_name.upper()} clustering failed for {layer_key} with params={params}: {e}")

    return plot_labels, labels_per_layer, color_maps


def cluster_analysis(args: argparse.Namespace, embeddings: dict, token_shapes: list, true_labels: np.ndarray, true_label_names: list,
                     metadata: pd.DataFrame=None, kinematics: pd.DataFrame=None, syllable_labels: pd.DataFrame=None,
                     data_type="raw", algorithms=["kmeans", "hdbscan", "gmm"]):

    output_dir = os.path.join(args.output_dir, data_type)
    fig_output_dir = os.path.join(output_dir, "figures")
    all_layers_output_dir = os.path.join(fig_output_dir, "all_layers")
    os.makedirs(fig_output_dir, exist_ok=True)
    os.makedirs(all_layers_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    list_of_embeddings, list_of_embeddings_fit, layer_flat_index_map = preprocess_for_cluster(embeddings, args)

    for alg in algorithms:
        plot_silhouette_scores(list_of_embeddings_fit, args, alg,
                               output_path=os.path.join(all_layers_output_dir, f"{alg}_silhouette_scores.png"))

    n_frames_to_plot = getattr(args, "n_frames_to_plot", 120000)
    plot_true_labels, plot_true_label_names = _resolve_labels(
        true_labels, true_label_names, metadata, syllable_labels,
        embeddings, layer_flat_index_map, token_shapes, args
    )

    alg_params = _build_alg_params(args, data_type, len(embeddings))
    clustroids_idxs = {}
    accumulators = {}  # {alg_name: (plot_labels, labels_per_layer, color_maps)}

    alg_titles = {
        "kmeans":  "Temporal Cluster Attribution Across Layers (KMeans)",
        "hdbscan": "Temporal Cluster Attribution Across Layers (HDBSCAN)",
        "gmm":     "Temporal Cluster Attribution Across Layers (GMM-Pyro)",
    }

    for alg_name in algorithms:
        title_print(f"Running {alg_name.upper()} Clustering Analysis",n=5)
        plot_labels, labels_per_layer, color_maps = cluster_all_layers(
            alg_name=alg_name, alg_params_per_layer=alg_params[alg_name], embeddings=embeddings, 
            list_of_embeddings=list_of_embeddings, list_of_embeddings_fit=list_of_embeddings_fit,
            layer_flat_index_map=layer_flat_index_map, token_shapes=token_shapes, clustroids_idxs=clustroids_idxs, 
            metadata=metadata, kinematics=kinematics,
            output_dir=output_dir, args=args
        )
        accumulators[alg_name] = (plot_labels, labels_per_layer, color_maps)
        os.makedirs(all_layers_output_dir, exist_ok=True)
        if plot_labels:
            plot_cluster_temporal(
                plot_labels, figsize=(18, 12), layer_labels=list(embeddings.keys()),
                true_label=plot_true_labels, true_label_names=plot_true_label_names,
                cluster_color_maps=color_maps,
                save_path=os.path.join(all_layers_output_dir, f"{alg_name}_temporal_clusters.png"),
                title=alg_titles[alg_name], n_frames_to_plot=n_frames_to_plot
            )

    return build_clustroid_full_map(
        clustroids_idxs, layer_flat_index_map, embeddings,
        *[accumulators[alg][1] for alg in ["kmeans", "hdbscan", "gmm"]]
    )

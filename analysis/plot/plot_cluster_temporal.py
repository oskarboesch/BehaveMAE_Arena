import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .plot_2D import get_category_palette
from analysis.utils.mari import mari as _mari


def _fit_label_count(labels, target_len, default_prefix):
    """Return labels with exactly target_len elements."""
    if labels is None:
        return [f"{default_prefix} {i}" for i in range(target_len)]

    labels = list(labels)
    if len(labels) >= target_len:
        return labels[:target_len]

    missing = target_len - len(labels)
    labels.extend([f"{default_prefix} {i}" for i in range(len(labels), len(labels) + missing)])
    return labels


def _cluster_color_map(labels, palette_name="tab10"):
    """Deterministic mapping from discrete cluster id to RGB color."""
    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))
    palette = plt.get_cmap(palette_name, max(1, len(unique_labels)))
    return {f"cluster {label}": palette(i)[:3] for i, label in enumerate(unique_labels)}


def _format_cluster_label(value):
    """Format cluster label into canonical string key used by color maps."""
    return f"cluster {value}"


def _labels_to_rgb_rows_with_maps(list_of_labels, num_frames, cluster_color_maps=None):
    """Convert per-layer 1D cluster labels into RGB rows using optional external color maps."""
    rgb_rows = []
    for layer_idx, layer_data in enumerate(list_of_labels):
        labels = np.asarray(layer_data)[:num_frames]
        if labels.ndim != 1:
            return None

        if cluster_color_maps is not None and layer_idx < len(cluster_color_maps) and cluster_color_maps[layer_idx] is not None:
            color_map = cluster_color_maps[layer_idx]
        else:
            color_map = _cluster_color_map(labels, palette_name="tab10")
        row_rgb = np.zeros((num_frames, 3), dtype=float)

        # Support both string keys ("cluster i") and numeric keys in color_map.
        if len(color_map) > 0 and isinstance(next(iter(color_map.keys())), str):
            label_keys = np.array([_format_cluster_label(v) for v in labels], dtype=object)
            for cluster_id, color in color_map.items():
                if isinstance(color, tuple) and len(color) == 2 and not isinstance(color[0], float):
                    color = color[0]
                row_rgb[:len(labels)][label_keys == cluster_id] = color
        else:
            for cluster_id, color in color_map.items():
                if isinstance(color, tuple) and len(color) == 2 and not isinstance(color[0], float):
                    color = color[0]
                row_rgb[:len(labels)][labels == cluster_id] = color
        rgb_rows.append(row_rgb)

    return np.stack(rgb_rows, axis=0)


def _flatten_true_labels(true_label_dict):
    """
    Flatten dict of true labels by run_id into a single (num_frames, num_labels) array.
    
    Args:
        true_label_dict: dict of {run_id: array shape (num_frames, num_labels)} with 0/1 values
    
    Returns:
        flattened array of shape (N, num_labels) concatenated across all runs
    """
    if not true_label_dict:
        return None
    
    arrays = list(true_label_dict.values())
    return np.concatenate(arrays, axis=0)


def _create_label_image(true_labels, num_frames, true_label_names=None):
    """
    Create RGB image from label array (num_frames, num_label_features).
    Maps categorical integer IDs to specific colors, with -1/NaN mapped to white.
    
    Args:
        true_labels: array shape (num_frames, num_features) with categorical integer values
        num_frames: number of frames to match with cluster data
        true_label_names: list of features for palette mapping
    
    Returns:
        RGB image array shape (num_features, num_frames, 3) with values in [0, 1]
    """
    if true_labels is None:
        return None
    
    # Crop to match num_frames
    true_labels_cropped = true_labels[:num_frames]
    num_features = true_labels_cropped.shape[1]
    
    label_img = np.ones((num_features, num_frames, 3))  # Default white
    
    for feature_idx in range(num_features):
        feature_data = true_labels_cropped[:, feature_idx]
        unique_vals = np.unique(feature_data[feature_data >= 0])
        num_unique = len(unique_vals)
        if num_unique == 0:
            continue
            
        category_name = true_label_names[feature_idx] if true_label_names and feature_idx < len(true_label_names) else ""
        palette_name = get_category_palette(category_name)
        palette = sns.color_palette(palette_name, n_colors=max(1, num_unique))
        
        for color_idx, val in enumerate(unique_vals):
            mask = feature_data == val
            label_img[feature_idx, mask, :] = palette[color_idx][:3]  # RGB only
    
    return label_img




def _extract_cluster_ids_per_layer(list_of_labels, num_frames):
    """
    Extract per-layer 1D cluster ids arrays from list_of_labels.
    Returns None if extraction is not possible.
    """
    cluster_ids = []
    for layer_data in list_of_labels:
        arr = np.asarray(layer_data)
        if arr.ndim != 1:
            return None
        cluster_ids.append(arr[:num_frames])
    return cluster_ids


def _compute_mari_matrix_all_to_all(true_labels, cluster_ids_per_layer, num_frames, true_label_names=None, max_signal_length=1000000):
    """
    Compute full all-to-all MARI matrix across:
    - all true label columns (one signal per label feature, arbitrary categorical values)
    - all layer clusterings (one signal per layer, cluster ids as full categorical vector)
 
    Each signal is a 1D integer array of length num_frames representing a full clustering.
    MARI is computed between every pair using the general K x L contingency table.
 
    Returns:
        mari_matrix: shape (n_signals, n_signals), diagonal is 1.0
        signal_names: list[str] length n_signals
    """
    signals = []
    signal_names = []
    if num_frames > max_signal_length:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(num_frames, size=max_signal_length, replace=False)
        print(f"Warning: num_frames={num_frames} exceeds max_signal_length={max_signal_length}, subsampling for MARI computation")
        true_labels = true_labels[indices]
        cluster_ids_per_layer = [ids[indices] for ids in cluster_ids_per_layer]
 
    if true_labels is not None:
        true_labels = true_labels[:num_frames]
        label_names = _fit_label_count(
            true_label_names,
            true_labels.shape[1],
            "Label",
        )
        for label_idx in range(true_labels.shape[1]):
            signals.append(true_labels[:, label_idx])
            signal_names.append(f"T:{label_names[label_idx]}")
 
    for layer_idx, cluster_ids in enumerate(cluster_ids_per_layer):
        # Each layer contributes a single full clustering vector — not per-cluster-value indicators
        signals.append(np.asarray(cluster_ids)[:num_frames])
        signal_names.append(f"L{layer_idx}")
 
    if len(signals) == 0:
        return None, None
 
    n_signals = len(signals)
    mari_matrix = np.eye(n_signals, dtype=float)  # diagonal = 1 by definition
    for i in tqdm(range(n_signals), desc="Computing MARI matrix", unit="pair"):
        for j in range(i + 1, n_signals):
            mari = _mari(signals[i], signals[j])
            mari_matrix[i, j] = mari
            mari_matrix[j, i] = mari
 
    return mari_matrix, signal_names
 
def plot_cluster_temporal(
    list_of_labels,
    figsize=(18, 36),
    layer_labels=None,
    true_label=None,
    true_label_names=None,
    list_of_cluster_ids=None,
    cluster_color_maps=None,
    title="Temporal Cluster Attribution Across Layers",
    save_path=None,
    n_frames_to_plot=None,
):
    num_layers = len(list_of_labels)

    # Keep originals for MARI computation (full resolution)
    list_of_labels_for_mari = list_of_labels
    true_label_for_mari = true_label
    list_of_cluster_ids_for_mari = list_of_cluster_ids


    # Subsample for plotting only
    if n_frames_to_plot is not None:
        list_of_labels = [lbl[:n_frames_to_plot] for lbl in list_of_labels]
        true_label = true_label['pseudo_run'][:n_frames_to_plot]


    num_frames = list_of_labels[0].shape[0]

    labels_img = _labels_to_rgb_rows_with_maps(
        list_of_labels,
        num_frames,
        cluster_color_maps=cluster_color_maps,
    )
    if labels_img is None:
        labels_img = np.stack(list_of_labels, axis=0)

    if isinstance(true_label, dict):
        true_label = _flatten_true_labels(true_label)

    label_img = _create_label_image(true_label, num_frames, true_label_names) if true_label is not None else None

    # MARI — full resolution, with length alignment
    mari_matrix, mari_signal_names = None, None
    if true_label_for_mari is not None:
        if isinstance(true_label_for_mari, dict):
            true_label_for_mari = _flatten_true_labels(true_label_for_mari)

        cluster_ids_per_layer = list_of_cluster_ids_for_mari
        if cluster_ids_per_layer is None:
            cluster_ids_per_layer = _extract_cluster_ids_per_layer(
                list_of_labels_for_mari, list_of_labels_for_mari[0].shape[0]
            )

        if cluster_ids_per_layer is not None:
            # Align all signals to minimum common length before MARI
            mari_num_frames = list_of_labels_for_mari[0].shape[0]
            min_len = min(
                mari_num_frames,
                true_label_for_mari.shape[0], # Correction : shape[0] au lieu de shape[-1]
                *[len(ids) for ids in cluster_ids_per_layer],
            )
            if min_len < mari_num_frames:
                print(f"MARI: truncating signals to common length {min_len} "
                      f"(was {mari_num_frames} frames, true_label {true_label_for_mari.shape[0]})")
            true_label_for_mari = true_label_for_mari[:min_len] # Correction : appliquer sur l'axe 0
            cluster_ids_per_layer = [ids[:min_len] for ids in cluster_ids_per_layer]

            mari_matrix, mari_signal_names = _compute_mari_matrix_all_to_all(
                true_label_for_mari,
                cluster_ids_per_layer,
                min_len,
                true_label_names=true_label_names,
            )

    # Create figure with subplots
    if label_img is not None and mari_matrix is not None:
        num_label_rows = label_img.shape[0]
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(figsize[0], figsize[1] + 4),
            gridspec_kw={"height_ratios": [num_layers, num_label_rows, max(2, num_label_rows)]},
        )
    elif label_img is not None:
        num_label_rows = label_img.shape[0]
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(figsize[0], figsize[1] + 2),
            gridspec_kw={"height_ratios": [num_layers, num_label_rows]},
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    ax1.imshow(labels_img, aspect="auto", interpolation="nearest")
    layer_labels = _fit_label_count(layer_labels, num_layers, "Layer")
    ax1.set_yticks(np.arange(num_layers))
    ax1.set_yticklabels(layer_labels)
    ax1.set_xlabel("Frame index")
    ax1.set_title(title)

    if label_img is not None:
        ax2.imshow(label_img, aspect="auto", interpolation="nearest")
        true_label_names = _fit_label_count(true_label_names, label_img.shape[0], "Label")
        ax2.set_yticks(np.arange(label_img.shape[0]))
        ax2.set_yticklabels(true_label_names)
        ax2.set_xlabel("Frame index")
        ax2.set_title("True Labels")

    if label_img is not None and mari_matrix is not None:
        im = ax3.imshow(mari_matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(mari_matrix.shape[0]):
            for j in range(mari_matrix.shape[1]):
                ax3.text(j, i, f"{mari_matrix[i, j]:.2f}", ha="center", va="center", color="black")
        ax3.set_title("MARI: Modified Adjusted Rand Index — Full All-to-All")
        ax3.set_ylabel("Signals")

        n_signals = mari_matrix.shape[0]
        tick_step = max(1, n_signals // 25)
        tick_idx = np.arange(0, n_signals, tick_step)
        ax3.set_yticks(tick_idx)
        if mari_signal_names is not None:
            ax3.set_yticklabels([mari_signal_names[i] for i in tick_idx])
        ax3.set_xticks(tick_idx)
        if mari_signal_names is not None:
            ax3.set_xticklabels([mari_signal_names[i] for i in tick_idx], rotation=90)
        ax3.set_xlabel("Signals")

        cbar = fig.colorbar(im, ax=ax3, fraction=0.015, pad=0.01)
        cbar.set_label("MARI")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        if mari_matrix is not None:
            mari_save_path = save_path.rsplit(".", 1)[0] + "_mari_matrix.csv"
            np.savetxt(mari_save_path, mari_matrix, delimiter=",", fmt="%.4f",
                       header=",".join(mari_signal_names), comments="")
    else:
        plt.show()
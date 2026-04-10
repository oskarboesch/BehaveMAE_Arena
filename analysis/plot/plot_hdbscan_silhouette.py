
from cuml.cluster import HDBSCAN 
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np


def _safe_silhouette_score(emb, labels, eps, layer_idx):
    """Return a safe silhouette score for potentially degenerate DBSCAN outputs."""
    emb = np.asarray(emb)
    labels = np.asarray(labels)

    n_samples = emb.shape[0]
    unique_labels = np.unique(labels)
    n_unique = len(unique_labels)

    # sklearn requires: 2 <= n_labels <= n_samples - 1
    if n_samples < 3:
        return 0.0

    if n_unique < 2 or n_unique >= n_samples:
        return 0.0

    # Each label must have at least 2 samples in practice for stable computation.
    _, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        return 0.0

    try:
        return float(silhouette_score(emb, labels))
    except ValueError as e:
        print(
            f"Warning: silhouette failed for layer {layer_idx}, eps={eps}: {e}. "
            "Using 0."
        )
        return 0.0

def plot_hdbscan_silhouettes(layers_embeddings, cluster_size_range=(10, 5000, 50), savepath=None):
    silhouette_scores = {i: [] for i in range(len(layers_embeddings))}
    n_clusters = {i: [] for i in range(len(layers_embeddings))}

    cluster_size_values = np.arange(cluster_size_range[0], cluster_size_range[1], cluster_size_range[2])

    for cluster_size in tqdm(cluster_size_values, desc="HDBSCAN Silhouette"):
        hdbscan = HDBSCAN(min_cluster_size=cluster_size, min_samples=cluster_size)
        for layer_idx, emb in enumerate(layers_embeddings):
            # ensure we have enough samples to apply HDBSCAN with this cluster_size value
            if emb.shape[0] < 50:  # HDBSCAN with less than 50 samples will likely label everything as noise, which is not informative
                print(f"Warning: not enough samples ({emb.shape[0]}) for HDBSCAN with cluster_size={cluster_size} for layer {layer_idx}. Skipping this cluster_size for this layer.")
                silhouette_scores[layer_idx].append(0)  # append 0 or some default value when we can't compute silhouette score
                n_clusters[layer_idx].append(0)  # append 0 clusters when we can't
                continue
            labels = hdbscan.fit_predict(emb)
            labels_np = np.asarray(labels)
            unique_labels = np.unique(labels_np)
            n_clusters[layer_idx].append(len(unique_labels) - (1 if -1 in unique_labels else 0))

            sil = _safe_silhouette_score(emb, labels_np, eps=cluster_size, layer_idx=layer_idx)
            silhouette_scores[layer_idx].append(sil)

    colors = sns.color_palette("tab10", n_colors=len(layers_embeddings))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # second y-axis for n_clusters since scales differ

    for i, color in enumerate(colors):
        sns.lineplot(
            x=cluster_size_values,
            y=silhouette_scores[i],
            label=f'Layer {i} silhouette',
            color=color,
            marker='o',
            markersize=6,
            linewidth=1.8,
            linestyle='-',
            ax=ax1,
        )
        sns.lineplot(
            x=cluster_size_values,
            y=n_clusters[i],
            label=f'Layer {i} n_clusters',
            color=color,
            marker='X',
            markersize=6,
            linewidth=1.4,
            linestyle='--',
            ax=ax2,
        )

    ax1.set_xlabel('HDBSCAN cluster size value')
    ax1.set_xticks(cluster_size_values)
    ax1.set_ylabel('Silhouette Score')
    ax2.set_ylabel('Number of Clusters')
    plt.title('Silhouette Scores and Cluster Counts for Each Layer (HDBSCAN)')
    ax1.grid(alpha=0.25, linestyle='--', linewidth=0.6)

    # combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
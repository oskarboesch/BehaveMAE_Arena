from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from .plot_hdbscan_silhouette import _safe_silhouette_score
import cupy as cp
import gc

from cuml.cluster import KMeans

def plot_k_means_silhouettes(layers_embeddings, k_range=(2, 11), random_state=42, savepath=None):
    K_range = range(k_range[0], k_range[1])

    # initialize scores per layer
    silhouette_scores = {i: [] for i in range(len(layers_embeddings))}

    for k in tqdm(K_range, desc="KMeans Silhouette"):
        for i, emb in enumerate(layers_embeddings):  # renamed loop var
            # ensure we have enough samples to apply KMeans with k clusters
            if emb.shape[0] < k:
                print(f"Warning: Not enough samples ({emb.shape[0]}) to apply KMeans with k={k} for layer {i}. Skipping this k for this layer.")
                silhouette_scores[i].append(0)  # append 0 or some default value when we can't compute silhouette score
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=random_state)
                labels = kmeans.fit_predict(emb)
                score = _safe_silhouette_score(emb, eps=None, labels=labels, layer_idx=i)
                silhouette_scores[i].append(score)
            except MemoryError:
                print(f"OOM for k={k}, layer {i} — skipping.")
                silhouette_scores[i].append(0)
            finally:
                if kmeans is not None:
                    del kmeans
                if labels is not None:
                    del labels
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                gc.collect()

    plt.figure(figsize=(10, 6))
    for i in range(len(layers_embeddings)):
        sns.lineplot(
            x=list(K_range),
            y=silhouette_scores[i],
            label=f'Layer {i}',
            marker='o',
            markersize=6,
            linewidth=1.8,
        )
    plt.xlabel('Number of clusters (k)')
    plt.xticks(K_range)
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Each Layer')
    plt.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
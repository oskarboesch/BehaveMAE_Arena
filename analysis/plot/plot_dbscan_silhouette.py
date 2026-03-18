
from cuml.cluster import DBSCAN 
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

def plot_dbscan_silhouettes(layers_embeddings, eps_range=(0.5, 5.0), min_samples=5, savepath=None):
    silhouette_scores = {i: [] for i in range(len(layers_embeddings))}
    n_clusters = {i: [] for i in range(len(layers_embeddings))}

    eps_values = np.arange(eps_range[0], eps_range[1], 0.5)

    for eps in tqdm(eps_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        for layer_idx, emb in enumerate(layers_embeddings):
            labels = dbscan.fit_predict(emb)
            n_clusters[layer_idx].append(len(set(labels)) - (1 if -1 in labels else 0))
            if len(set(labels)) > 1:
                silhouette_scores[layer_idx].append(silhouette_score(emb, labels))
            else:
                silhouette_scores[layer_idx].append(0)

    colors = sns.color_palette("tab10", n_colors=len(layers_embeddings))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # second y-axis for n_clusters since scales differ

    for i, color in enumerate(colors):
        ax1.plot(eps_values, silhouette_scores[i], label=f'Layer {i} silhouette',
                 color=color, marker='o', linestyle='-')
        ax2.plot(eps_values, n_clusters[i], label=f'Layer {i} n_clusters',
                 color=color, marker='x', linestyle='--')

    ax1.set_xlabel('DBSCAN eps value')
    ax1.set_xticks(eps_values)
    ax1.set_ylabel('Silhouette Score')
    ax2.set_ylabel('Number of Clusters')
    plt.title('Silhouette Scores and Cluster Counts for Each Layer (DBSCAN)')

    # combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
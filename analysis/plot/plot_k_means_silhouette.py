from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

from cuml.cluster import KMeans

def plot_k_means_silhouettes(layers_embeddings, k_range=(2, 11), random_state=42, savepath=None):
    K_range = range(k_range[0], k_range[1])

    # initialize scores per layer
    silhouette_scores = {i: [] for i in range(len(layers_embeddings))}

    for k in tqdm(K_range):
        for i, emb in enumerate(layers_embeddings):  # renamed loop var
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(emb)
            silhouette_scores[i].append(silhouette_score(emb, labels))

    plt.figure(figsize=(10, 6))
    for i in range(len(layers_embeddings)):
        plt.plot(K_range, silhouette_scores[i], label=f'Layer {i}')
    plt.xlabel('Number of clusters (k)')
    plt.xticks(K_range)
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Each Layer')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
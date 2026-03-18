import numpy as np
from cuml.cluster import KMeans, DBSCAN 
from .plot.plot_k_means_silhouette import plot_k_means_silhouettes
from .plot.plot_dbscan_silhouette import plot_dbscan_silhouettes
from .plot.plot_2D import plot_2D
from tqdm import tqdm
import os

def cluster(args, embeddings, data_type="raw"):

    list_of_embeddings = []
    list_of_sampled_embeddings = []

    for layer_key, embeddings_dict in tqdm(embeddings.items(), desc="Preparing embeddings for clustering"):
        # flatten the embeddings across all runs 
        sampled_X = np.concatenate([v[::args.silhouette_ds_rate] for v in embeddings_dict.values()], axis=0)
        X = np.concatenate(list(embeddings_dict.values()), axis=0)
        list_of_embeddings.append(X)
        list_of_sampled_embeddings.append(sampled_X)

    plot_k_means_silhouettes(list_of_sampled_embeddings, k_range=args.kmeans_k_range, random_state=42, savepath=f"{args.output_dir}/kmeans_silhouette_scores.png")
    plot_dbscan_silhouettes(list_of_sampled_embeddings, eps_range=args.dbscan_eps_range, min_samples=args.dbscan_min_samples, savepath=f"{args.output_dir}/dbscan_silhouette_scores.png")

    for layer_idx, layer_key in enumerate(embeddings.keys()):
        output_path = f"{args.output_dir}/{data_type}/{layer_key}"
        os.makedirs(output_path, exist_ok=True)
        if len(args.kmeans_ks) > layer_idx:
            k = args.kmeans_ks[layer_idx]
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(list_of_embeddings[layer_idx])
            np.save(os.path.join(output_path, f"kmeans_labels_k_{k}.npy"), labels)
            plot_2D(list_of_embeddings[layer_idx], labels, output_path=f"{output_path}/kmeans_plot_k_{k}.png")

        if len(args.dbscan_eps) > layer_idx:
            eps = args.dbscan_eps[layer_idx]
            dbscan = DBSCAN(eps=eps, min_samples=args.dbscan_min_samples)
            dbscan_labels = dbscan.fit_predict(list_of_embeddings[layer_idx])
            np.save(os.path.join(output_path, f"dbscan_labels_eps_{eps}_min_samples_{args.dbscan_min_samples}.npy"), dbscan_labels)
            plot_2D(list_of_embeddings[layer_idx], dbscan_labels, output_path=f"{output_path}/dbscan_plot_eps_{eps}_min_samples_{args.dbscan_min_samples}.png")
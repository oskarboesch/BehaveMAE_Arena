import numpy as np
from cuml.cluster import KMeans, DBSCAN 
from util.plot_utils import plot_dbscan_silhouettes, plot_k_means_silhouettes

def main():
    run_id = ''
    experiment = 'experiment1'  # Change this to experiment2 for the second set of embeddings
    n_samples_per_video = 10**3  # Number of samples to use for silhouette score calculation (for large datasets)
    output_path = f"outputs/arena/{experiment}"
    dict_layer0_path = f"{output_path}/test_submission_0.npy"
    dict_layer1_path = f"{output_path}/test_submission_1.npy"
    dict_layer2_path = f"{output_path}/test_submission_2.npy"

    dict_layer0 = np.load(dict_layer0_path, allow_pickle=True).item()
    dict_layer1 = np.load(dict_layer1_path, allow_pickle=True).item()
    dict_layer2 = np.load(dict_layer2_path, allow_pickle=True).item()

    # sample randomly n_samples_per_video idx for each frame_idx elements
    kept_idx=[]
    if run_id is None:
        frame_number_map = dict_layer0['frame_number_map'].keys()
    else :
        # take the dictionary at position run_id
        frame_number_map= [dict_layer0['frame_number_map'][run_id]]

    for frame_idx in frame_number_map:
        idx = dict_layer0['frame_number_map'][frame_idx]
        idxs = np.arange(idx[0], idx[1])
        if len(idxs) > n_samples_per_video:
            kept_idx.extend(np.random.choice(idxs, size=n_samples_per_video, replace=False))
        else:
            kept_idx.extend(idxs)

    embeddings_layer0 = dict_layer0['embeddings']
    embeddings_layer1 = dict_layer1['embeddings']
    embeddings_layer2 = dict_layer2['embeddings']

    sampled_list_of_embeddings = [embeddings_layer0[kept_idx], embeddings_layer1[kept_idx], embeddings_layer2[kept_idx]]
    list_of_embeddings = [embeddings_layer0, embeddings_layer1, embeddings_layer2]

    # plot_k_means_silhouettes(sampled_list_of_embeddings, savepath=f"{output_path}/kmeans_silhouette_scores.png")
    # plot_dbscan_silhouettes(sampled_list_of_embeddings, eps_range=(0.5, 5.0), min_samples=5, savepath=f"{output_path}/dbscan_silhouette_scores.png")

    kmeans_layer0 = KMeans(n_clusters=2, random_state=0).fit(embeddings_layer0)
    kmeans_layer1 = KMeans(n_clusters=2, random_state=0).fit(embeddings_layer1)
    kmeans_layer2 = KMeans(n_clusters=2, random_state=0).fit(embeddings_layer2)

    dbscan_layer0 = DBSCAN(eps=1.5, min_samples=10).fit(embeddings_layer0)
    dbscan_layer1 = DBSCAN(eps=4.5, min_samples=10).fit(embeddings_layer1)
    dbscan_layer2 = DBSCAN(eps=0.5, min_samples=10).fit(embeddings_layer2)

    # save the cluster labels for later analysis
    np.save(f"{output_path}/kmeans_labels_layer0.npy", kmeans_layer0.labels_)
    np.save(f"{output_path}/kmeans_labels_layer1.npy", kmeans_layer1.labels_)
    np.save(f"{output_path}/kmeans_labels_layer2.npy", kmeans_layer2.labels_)

    np.save(f"{output_path}/dbscan_labels_layer0.npy", dbscan_layer0.labels_)
    np.save(f"{output_path}/dbscan_labels_layer1.npy", dbscan_layer1.labels_)
    np.save(f"{output_path}/dbscan_labels_layer2.npy", dbscan_layer2.labels_)


if __name__ == "__main__":
    main()
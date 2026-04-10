
import numpy as np
def get_clustroid_idx(embeddings, cluster_labels, cluster_id):
    """
    Get the clustroid (most central point) of a cluster given the embeddings and cluster labels.
    """
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        print(f"No points found for cluster {cluster_id}.")
        return None
    cluster_embeddings = embeddings[cluster_indices]
    centroid = np.mean(cluster_embeddings, axis=0)
    
    # Compute distances to centroid and find the index of the closest point
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    clustroid_index = cluster_indices[np.argmin(distances)]
    
    return clustroid_index
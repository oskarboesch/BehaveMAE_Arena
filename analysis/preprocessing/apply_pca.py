from sklearn.decomposition import PCA
from analysis.plot.plot_pca import plot_pca_explained_variance
import numpy as np
import os

def apply_pca(embeddings, n_components=64, output_dir=None):

    for layer_key in embeddings.keys():
        # apply pca if the embedding dimension is larger than n_components
        if embeddings[layer_key][next(iter(embeddings[layer_key]))].shape[1] > n_components:
            print(f"Applying PCA to layer {layer_key} with original dimension {embeddings[layer_key][next(iter(embeddings[layer_key]))].shape[1]}")
            # concatenate all runs for this layer
            all_embeddings = np.concatenate(list(embeddings[layer_key].values()), axis=0) # shape [num_frames_total, embedding_dim]
            pca = PCA(n_components=n_components)
            all_embeddings_pca = pca.fit_transform(all_embeddings)  # shape [num_frames_total, n_components]
            # split back into runs
            start_idx = 0
            for run_id in embeddings[layer_key].keys():
                num_frames = embeddings[layer_key][run_id].shape[0]
                embeddings[layer_key][run_id] = all_embeddings_pca[start_idx:start_idx+num_frames] 
                start_idx += num_frames
            print(f"Reduced dimension to {n_components} for layer {layer_key}")
            if output_dir is not None:
                dir = os.path.join(output_dir, "pca")
                os.makedirs(dir, exist_ok=True)
                plot_pca_explained_variance(pca, n_components=n_components, save_path=os.path.join(dir, f"pca_explained_variance_{layer_key}.png"))
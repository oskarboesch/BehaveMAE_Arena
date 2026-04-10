from sklearn.decomposition import PCA, IncrementalPCA
from analysis.plot.plot_pca import plot_pca_explained_variance
import numpy as np
import os

def _iter_run_batches(run_array, batch_size):
    n_samples = run_array.shape[0]
    for start in range(0, n_samples, batch_size):
        yield run_array[start:start + batch_size]


def apply_pca(embeddings, n_components=64, output_dir=None, batch_size=50000, seed=42):

    for layer_key in embeddings.keys():
        first_run = next(iter(embeddings[layer_key]))
        original_dim = embeddings[layer_key][first_run].shape[1]

        # apply pca if the embedding dimension is larger than n_components
        if original_dim > n_components:
            # check we have enough samples to apply PCA
            total_samples = sum(run_emb.shape[0] for run_emb in embeddings[layer_key].values())
            if total_samples < n_components:
                print(f"Warning: Not enough samples ({total_samples}) to apply PCA with n_components={n_components} for layer {layer_key}. Skipping PCA for this layer.")
                continue
            print(f"Applying PCA to layer {layer_key} with original dimension {embeddings[layer_key][next(iter(embeddings[layer_key]))].shape[1]}")

            # Use IncrementalPCA for large datasets to avoid materializing all frames in memory.
            if total_samples > batch_size:
                pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
                buffer = []
                buffer_size = 0
                for run_emb in embeddings[layer_key].values():
                    for chunk in _iter_run_batches(run_emb, batch_size):
                        buffer.append(chunk)
                        buffer_size += chunk.shape[0]
                        if buffer_size >= n_components:
                            pca.partial_fit(np.concatenate(buffer, axis=0))
                            buffer = []
                            buffer_size = 0
                if buffer:  # flush remaining
                    combined = np.concatenate(buffer, axis=0)
                    if combined.shape[0] >= n_components:
                        pca.partial_fit(combined)
                # Second pass: transform each run in chunks and replace in-place.
                for run_id, run_emb in embeddings[layer_key].items():
                    transformed_chunks = [pca.transform(chunk) for chunk in _iter_run_batches(run_emb, batch_size)]
                    embeddings[layer_key][run_id] = np.concatenate(transformed_chunks, axis=0)
            else:
                # Small enough to run standard PCA efficiently.
                all_embeddings = np.concatenate(list(embeddings[layer_key].values()), axis=0)
                pca = PCA(n_components=n_components, random_state=seed)
                all_embeddings_pca = pca.fit_transform(all_embeddings)  # shape [num_frames_total, n_components]

                # Split transformed matrix back per run.
                start_idx = 0
                for run_id in embeddings[layer_key].keys():
                    num_frames = embeddings[layer_key][run_id].shape[0]
                    embeddings[layer_key][run_id] = all_embeddings_pca[start_idx:start_idx + num_frames]
                    start_idx += num_frames

            print(f"Reduced dimension to {n_components} for layer {layer_key}")
            if output_dir is not None:
                dir = os.path.join(output_dir, "pca")
                os.makedirs(dir, exist_ok=True)
                plot_pca_explained_variance(pca, n_components=n_components, save_path=os.path.join(dir, f"pca_explained_variance_{layer_key}.png"))
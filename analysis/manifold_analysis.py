from tqdm import tqdm
import numpy as np
import os
from .utils.window_and_aggregate import window_and_aggregate

def manifold_analysis(embeddings, args, token_shapes):
    from cuml.manifold import TSNE, UMAP
    tsne_embeddings = {}
    umap_embeddings = {}
    
    for layer_key, embeddings_dict in tqdm(embeddings.items(), desc="Manifold Processing layers"):
        X = np.concatenate(list(embeddings_dict.values()), axis=0)

        # check that we have enough samples to run tsne/umap
        if X.shape[0] < 10:
            print(f"Not enough samples to run TSNE/UMAP for {layer_key} (n_samples={X.shape[0]}), skipping manifold analysis for this layer.")
            continue
        
        print(f"Running TSNE and UMAP for {layer_key} with {X.shape[0]} samples and {X.shape[1]} dimensions.")
        tsne = TSNE(n_components=2, random_state=args.seed).fit_transform(X)
        umap = UMAP(n_components=2, random_state=args.seed).fit_transform(X)
        
        # rebuild the mapping from run_id to tsne/umap embeddings based on the window_map
        tsne_dict = {}
        umap_dict = {}
        offset = 0
        for run_id, emb in embeddings_dict.items():
            n_windows = emb.shape[0]
            tsne_dict[run_id] = tsne[offset:offset + n_windows]
            umap_dict[run_id] = umap[offset:offset + n_windows]
            offset += n_windows

        np.save(os.path.join(args.output_dir, f"{layer_key}_tsne.npy"), tsne_dict)
        np.save(os.path.join(args.output_dir, f"{layer_key}_umap.npy"), umap_dict)
        print(f"Saved manifold embeddings for {layer_key} to {args.output_dir}")
        tsne_embeddings[layer_key] = tsne_dict
        umap_embeddings[layer_key] = umap_dict


    return tsne_embeddings, umap_embeddings
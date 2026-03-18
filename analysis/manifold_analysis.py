from tqdm import tqdm
import numpy as np
import os
from .utils.window_and_aggregate import window_and_aggregate

def manifold_analysis(embeddings, args, token_shapes):
    from cuml.manifold import TSNE, UMAP
    tsne_embeddings = {}
    umap_embeddings = {}
    layer_window_maps = {}
    
    for layer_key, embeddings_dict in tqdm(embeddings.items(), desc="Manifold Processing layers"):
        layer_window = max(1, args.window_size_manifold // token_shapes[int(layer_key.split("_")[-1])][0])  # convert window size from frames to tokens
        layer_stride = max(1, args.window_stride_manifold // token_shapes[int(layer_key.split("_")[-1])][0])  # convert stride from frames to tokens
        embeddings_windowed, window_map = window_and_aggregate(embeddings_dict, window_size=layer_window, stride=layer_stride, method=args.agg_method)
        X = np.concatenate(list(embeddings_windowed.values()), axis=0)

        tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
        umap = UMAP(n_components=2, random_state=0).fit_transform(X)
        
        # rebuild the mapping from run_id to tsne/umap embeddings based on the window_map
        tsne_dict = {}
        umap_dict = {}
        offset = 0
        for run_id, emb in embeddings_windowed.items():
            n_windows = emb.shape[0]
            tsne_dict[run_id] = tsne[offset:offset + n_windows]
            umap_dict[run_id] = umap[offset:offset + n_windows]
            offset += n_windows

        np.save(os.path.join(args.output_dir, f"{layer_key}_tsne.npy"), tsne_dict)
        np.save(os.path.join(args.output_dir, f"{layer_key}_umap.npy"), umap_dict)
        print(f"Saved manifold embeddings for {layer_key} to {args.output_dir}")
        tsne_embeddings[layer_key] = tsne_dict
        umap_embeddings[layer_key] = umap_dict
        layer_window_maps[layer_key] = window_map


    return tsne_embeddings, umap_embeddings, layer_window_maps
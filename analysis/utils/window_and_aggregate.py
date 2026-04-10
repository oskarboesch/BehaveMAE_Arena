import numpy as np

def window_and_aggregate(embeddings_dict, window_size=1, stride=1, method="mean", n_windows_per_run=-1, seed=None):
    windowed_dict = {}
    window_map = {}  # {run_id: array of original token indices that each window corresponds to}
    rng = np.random.default_rng(seed) if seed is not None else None

    for run_id, emb in embeddings_dict.items():
        n_tokens, emb_dim = emb.shape

        if window_size == 1 and stride == 1:
            indices = np.arange(n_tokens)
            if n_windows_per_run != -1 and len(indices) > n_windows_per_run:
                if rng is None:
                    indices = np.random.choice(indices, size=n_windows_per_run, replace=False)
                else:
                    indices = rng.choice(indices, size=n_windows_per_run, replace=False)
                indices = np.sort(indices)
            if len(indices) == 0:
                continue
            windowed_dict[run_id] = emb[indices]
            window_map[run_id] = indices
            continue
        indices = np.arange(0, n_tokens - window_size + 1, stride)
        if len(indices) == 0:
            continue

        if n_windows_per_run != -1:
            # if there are more windows than n_windows_per_run, sample a subset of windows
            if len(indices) > n_windows_per_run:
                if rng is None:
                    indices = np.random.choice(indices, size=n_windows_per_run, replace=False)
                else:
                    indices = rng.choice(indices, size=n_windows_per_run, replace=False)
                indices = np.sort(indices)  # sort the indices to maintain temporal order

        windows = np.stack([emb[i:i + window_size] for i in indices])

        if method == "mean":
            aggregated = windows.mean(axis=1)
        elif method == "max":
            aggregated = windows.max(axis=1)
        elif method == "first":
            aggregated = windows[:, 0, :]
        elif method == "last":
            aggregated = windows[:, -1, :]
        windowed_dict[run_id] = aggregated
        window_map[run_id] = indices + window_size // 2  # center token of each window → used to index labels
    print(f"Applied windowing and aggregation with method={method}, window_size={window_size}, stride={stride}, n_windows_per_run={n_windows_per_run}.")
    return windowed_dict, window_map

def window_and_aggregate_all_layers(embeddings, token_shapes, window_size=1, stride=1, agg_method="mean", n_windows_per_run=-1, seed=None):
    embeddings_windowed = {}
    raw_layer_window_maps = {}
    for layer_key, embeddings_dict in embeddings.items():
        token_shape = token_shapes[int(layer_key.split("_")[-1])]
        w_s = max(1, window_size // token_shape[0])
        w_str = max(1, stride // token_shape[0])
        print(f"Windowing layer {layer_key} with window size {w_s} and stride {w_str} (token shape: {token_shape})")
        embedding_windowed, layer_window_map = window_and_aggregate(embeddings_dict, window_size=w_s, 
                                                            stride=w_str, method=agg_method, n_windows_per_run=n_windows_per_run, seed=seed)
        embeddings_windowed[layer_key] = embedding_windowed
        raw_layer_window_maps[layer_key] = layer_window_map
    windowed_token_shapes = [(max(window_size, token_shapes[i][0]) if token_shapes[i][0] > 0 else token_shapes[i][0],  token_shapes[i][1],  token_shapes[i][2]) for i in range(len(token_shapes))]  # after windowing all embeddings have the same token shape equal to the base window size
    print(f"Resulting windowed token shapes: {windowed_token_shapes}")
    return embeddings_windowed, raw_layer_window_maps, windowed_token_shapes
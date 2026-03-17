import numpy as np

def window_and_aggregate(embeddings_dict, window_size=1, stride=1, method="mean"):
    windowed_dict = {}
    window_map = {}  # {run_id: array of original token indices that each window corresponds to}

    for run_id, emb in embeddings_dict.items():
        n_tokens, emb_dim = emb.shape

        if window_size == 1 and stride == 1:
            windowed_dict[run_id] = emb
            window_map[run_id] = np.arange(n_tokens)
            continue

        indices = np.arange(0, n_tokens - window_size + 1, stride)
        if len(indices) == 0:
            continue

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

    return windowed_dict, window_map
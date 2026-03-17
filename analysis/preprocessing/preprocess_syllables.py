from scipy.stats import mode
from analysis.utils.window_and_aggregate import window_and_aggregate
from tqdm import tqdm
import numpy as np

def preprocess_syllables(embeddings_dict, syllable_labels, metadata, token_shape, args, window_map):
    if window_map is None:
        num_frames_per_token = token_shape[0]
    else:
        num_frames_per_token = args.window_size_manifold * token_shape[0]

    # Apply syllable-specific windowing on top of whatever was already applied
    layer_window = max(1, args.window_size_syllables // num_frames_per_token)
    layer_stride = max(1, args.window_stride_syllables // num_frames_per_token)

    embeddings_windowed, syllable_window_map = window_and_aggregate(
        embeddings_dict,
        window_size=layer_window,
        stride=layer_stride,
        method=args.agg_method
    )

    X = []
    y = []
    groups = []

    for run_id, emb in tqdm(embeddings_windowed.items(), desc="Preprocessing syllable labels"):  # iterate over windowed embeddings
        if run_id not in syllable_labels:
            print(f"No syllable labels found for run_id {run_id}, skipping.")
            continue
        run_syllables = syllable_labels[run_id]  # shape (n_frames,)
        run_metadata = metadata[metadata["run_id"] == run_id]

        # Reshape into (n_tokens, num_frames_per_token)
        max_frames = len(run_syllables)
        n_tokens = (max_frames - num_frames_per_token) // num_frames_per_token + 1
        if n_tokens <= 0:
            print(f"Not enough frames ({max_frames}) for run_id {run_id} to form even one window of {num_frames_per_token}, skipping.")
            continue
        slice_end = n_tokens * num_frames_per_token

        syll_windows = run_syllables[:slice_end].reshape(n_tokens, num_frames_per_token)

        # NaN fraction per token → shape (n_tokens,)
        nan_frac = np.isnan(syll_windows).mean(axis=1)
        valid_mask = nan_frac <= args.max_nan_ratio_per_window

        # Compute mode over the window dimension → (n_tokens,)
        with np.errstate(all='ignore'):
            syll_modes, _ = mode(syll_windows, axis=1, nan_policy='omit')
        syll_modes = syll_modes.flatten()

        # Use syllable_window_map (not the manifold window_map) to align embedding rows to token indices
        token_indices = syllable_window_map[run_id] if run_id in syllable_window_map else np.arange(emb.shape[0])

        # Only keep embedding rows whose center token is valid (in bounds + not NaN-heavy)
        in_bounds = token_indices < n_tokens
        clamped_indices = np.clip(token_indices, 0, n_tokens - 1)
        valid_emb_mask = in_bounds & valid_mask[clamped_indices]

        if not valid_emb_mask.any():
            continue

        valid_token_indices = token_indices[valid_emb_mask]
        X.append(emb[valid_emb_mask])
        y.append(syll_modes[valid_token_indices])
        groups.append(np.full(valid_emb_mask.sum(), run_metadata.iloc[0]["animal_id"]))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    groups = np.concatenate(groups, axis=0)

    if len(X) > args.max_n_samples_for_syllable_analysis:
        indices = np.random.choice(len(X), args.max_n_samples_for_syllable_analysis, replace=False)
        X = X[indices]
        y = y[indices]
        groups = groups[indices]
    print(f"Prepared {len(X)} tokens with syllable labels")
    return X, y, groups
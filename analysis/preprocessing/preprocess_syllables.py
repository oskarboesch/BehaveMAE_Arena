from scipy.stats import mode
from tqdm import tqdm
import numpy as np

def preprocess_syllables(embeddings_dict, syllable_labels, metadata, windowed_token_shape, raw_token_shape, args, window_map):

    num_frames_per_token = windowed_token_shape[0]

    X = []
    y = []
    groups = []

    for run_id, emb in tqdm(embeddings_dict.items(), desc="Preprocessing syllable labels"):
        if run_id not in syllable_labels:
            print(f"No syllable labels found for run_id {run_id}, skipping.")
            continue
        run_syllables = syllable_labels[run_id]  # shape (n_frames,)
        run_metadata = metadata[metadata["run_id"] == run_id]

        n_tokens = emb.shape[0]
        n_kin_tokens = np.ceil(run_syllables.shape[0] / num_frames_per_token).astype(int)
        if n_tokens > n_kin_tokens:
            n_kin_tokens = n_tokens
        max_frames = n_kin_tokens * num_frames_per_token

        # Pad at end (not start) and slice to max_frames
        padded = np.pad(run_syllables, (0, max(0, max_frames - len(run_syllables))), mode='edge')
        syll_windows = padded.reshape(n_kin_tokens, num_frames_per_token)

        # NaN fraction per token → shape (n_tokens,)
        nan_frac = np.isnan(syll_windows.astype(float)).mean(axis=1)
        valid_mask = nan_frac <= args.max_nan_ratio_per_window

        # Compute mode over the window dimension → (n_tokens,)
        with np.errstate(all='ignore'):
            syll_modes, _ = mode(syll_windows, axis=1, nan_policy='omit')
        syll_modes = syll_modes.flatten()

        # Convert frame indices to token indices
        num_tokens_per_window = num_frames_per_token // raw_token_shape[0]
        window_indices = window_map[run_id] // num_tokens_per_window if run_id in window_map else np.arange(n_tokens)

        in_bounds = window_indices < len(valid_mask) # we need to do this because we padded the input sequence to infer the embeddings
        window_indices = window_indices[in_bounds]
        emb = emb[in_bounds]
        # Only keep embedding rows whose center token is valid (not NaN-heavy)
        valid_emb_mask = valid_mask[window_indices]

        if not valid_emb_mask.any():
            print(f"No valid tokens for run {run_id}, skipping.")
            continue

        valid_token_indices = window_indices[valid_emb_mask]
        X.append(emb[valid_emb_mask])
        y.append(syll_modes[valid_token_indices])
        groups.append(np.full(valid_emb_mask.sum(), run_metadata.iloc[0]["animal_id"]))

    if not X:
        raise ValueError("No valid tokens found across all runs.")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    groups = np.concatenate(groups, axis=0)

    if len(X) > args.max_n_samples_for_syllable_analysis:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(X), args.max_n_samples_for_syllable_analysis, replace=False)
        X = X[indices]
        y = y[indices]
        groups = groups[indices]

    print(f"Prepared {len(X)} tokens with syllable labels")
    return X, y, groups
import numpy as np
from tqdm import tqdm

def preprocess_kinematics(embeddings_dict, kinematics, windowed_token_shape, raw_token_shape, kin_vars, args, window_map):
    X = []
    y = []

    num_frames_per_token = windowed_token_shape[0]

    for run_id, emb in tqdm(embeddings_dict.items(), desc="Preprocessing kinematics"):
        if run_id not in kinematics:
            print(f"No kinematics found for run_id {run_id}, skipping.")
            continue
        run_kinematics = kinematics[run_id]

        if num_frames_per_token == -1:
            num_frames_per_token = run_kinematics[kin_vars[0]].shape[0]

        n_tokens = emb.shape[0]
        n_kin_tokens = np.ceil(run_kinematics[kin_vars[0]].shape[0] / num_frames_per_token).astype(int)
        if n_tokens > n_kin_tokens:
            n_kin_tokens = n_tokens

        max_frames = n_kin_tokens * num_frames_per_token

        # Stack all kin_vars into (n_kin_vars, max_frames)
        kin_matrix = np.stack([
            np.pad(run_kinematics[kv], (0, max(0, max_frames - len(run_kinematics[kv]))), mode='edge')
            for kv in kin_vars
        ])

        # Reshape into (n_kin_vars, n_tokens, num_frames_per_token)
        kin_windows = kin_matrix.reshape(len(kin_vars), n_kin_tokens, num_frames_per_token)

        # NaN fraction per (kin_var, token) → shape (n_kin_vars, n_tokens)
        nan_frac = np.isnan(kin_windows).mean(axis=2)
        valid_mask = (nan_frac <= args.max_nan_ratio_per_window).all(axis=0)  # (n_tokens,)

        # Compute nanmean over window dimension → (n_kin_vars, n_tokens)
        with np.errstate(all='ignore'):
            kin_means = np.nanmean(kin_windows, axis=2)

        # Map run embeddings to token indices
        num_tokens_per_window = num_frames_per_token // raw_token_shape[0]
        window_indices = window_map[run_id] // num_tokens_per_window if window_map is not None else np.arange(n_kin_tokens)
        in_bounds = window_indices < len(valid_mask) # we need to do this because we padded the input sequence to infer the embeddings
        window_indices = window_indices[in_bounds]
        emb = emb[in_bounds]

        valid_emb_mask = valid_mask[window_indices]
        if not valid_emb_mask.any():
            print(f"No valid tokens for run {run_id}, skipping.")
            continue

        X.append(emb[valid_emb_mask])                      # (n_valid, emb_dim)
        y.append(kin_means[:, window_indices[valid_emb_mask]].T)       # (n_valid, n_kin_vars)

    if not X:
        raise ValueError("No valid tokens found across all runs.")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print(f"Prepared {len(X)} tokens, {X.shape[1]} embedding dims, {y.shape[1]} kinematic variables")
    return X, y
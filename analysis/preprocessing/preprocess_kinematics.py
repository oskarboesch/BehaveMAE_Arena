import numpy as np
from tqdm import tqdm

def preprocess_kinematics(embeddings_dict, kinematics, token_shape, kin_vars, args, window_map):
    X = []
    y = []

    num_frames_per_token = token_shape[0]

    for run_id, emb in tqdm(embeddings_dict.items(), desc="Preprocessing kinematics"):
        if run_id not in kinematics:
            print(f"No kinematics found for run_id {run_id}, skipping.")
            continue
        run_kinematics = kinematics[run_id]

        # Stack all kin_vars into a (n_kin_vars, n_frames) array once
        max_frames = max(len(run_kinematics[kv]) for kv in kin_vars)
        n_tokens = (max_frames - num_frames_per_token) // num_frames_per_token + 1
        slice_end = n_tokens * num_frames_per_token

        kin_matrix = np.stack([
            np.pad(run_kinematics[kv], (max_frames - len(run_kinematics[kv]), 0), mode='edge')[:slice_end]
            for kv in kin_vars
        ])

        # Reshape into (n_kin_vars, n_tokens, num_frames_per_token)
        kin_windows = kin_matrix.reshape(len(kin_vars), n_tokens, num_frames_per_token)

        # NaN fraction per (kin_var, token) → shape (n_kin_vars, n_tokens)
        nan_frac = np.isnan(kin_windows).mean(axis=2)
        valid_mask = (nan_frac <= args.max_nan_ratio_per_window).all(axis=0)  # shape (n_tokens,)

        # Compute nanmean over the window dimension → (n_kin_vars, n_tokens)
        with np.errstate(all='ignore'):
            kin_means = np.nanmean(kin_windows, axis=2)

        # Use window_map to align: each row in emb corresponds to a center token index
        token_indices = window_map[run_id] if window_map is not None else np.arange(emb.shape[0])

        # Only keep embedding rows whose center token is valid (within bounds + not NaN-heavy)
        in_bounds = token_indices < n_tokens
        valid_emb_mask = in_bounds & valid_mask[np.where(in_bounds, token_indices, 0)]

        if not valid_emb_mask.any():
            continue

        valid_token_indices = token_indices[valid_emb_mask]   # which tokens to pull kinematics from
        X.append(emb[valid_emb_mask])                         # (n_valid, emb_dim)
        y.append(kin_means[:, valid_token_indices].T)         # (n_valid, n_kin_vars)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print(f"Prepared {len(X)} tokens, {X.shape[1]} embedding dims, {y.shape[1]} kinematic variables")
    return X, y
from analysis.utils.window_and_aggregate import window_and_aggregate
import numpy as np

def preprocess_metadata(args, embeddings_dict, metadata, meta_var, token_shape):    

    X = []
    X_plot = []  # for plotting only, not used in modeling
    y = []
    y_plot = []  # for plotting only, not used in modeling
    groups = []
    if args.n_windows_per_run_for_metadata == -1:
        min_n_windows = min(emb.shape[0] for emb in embeddings_dict.values())
        print(f"Using the minimum number of windows across all runs for metadata classification: {min_n_windows}")


    for run_id, emb in embeddings_dict.items():  # iterate over windowed embeddings
        run_metadata = metadata[metadata["run_id"] == run_id]
        if len(run_metadata) == 0:
            print(f"No metadata found for run_id {run_id}, skipping.")
            continue
        elif len(run_metadata) > 1:
            print(f"Multiple metadata entries found for run_id {run_id}, using the first one.")
        meta_value = run_metadata.iloc[0][meta_var]
        if meta_value == "Unknown":
            continue
        animal_id = run_metadata.iloc[0]["animal_id"]
        n_windows = len(emb)
        # sample randomly args.n_windows windows 
        if n_windows < args.n_windows_per_run_for_metadata:
            if token_shape[0] == -1:
                n_windows_to_sample = 1 # if per sequence embedding we have only one "window" per run, so we set n_windows_per_run_for_metadata to 1 to avoid errors, but this also means we won't be able to sample multiple windows for metadata classification, which may limit the performance of the metadata classification. Consider adjusting the window size/stride or using a different embedding method that allows for more windows per run.
            else:
                print(f"Not enough windows ({n_windows}) for run_id {run_id} to sample {args.n_windows_per_run_for_metadata} windows for metadata classification. Consider reducing the number of windows per run or adjusting the window size/stride.")
                continue
        else :
            if args.n_windows_per_run_for_metadata == -1:
                n_windows_to_sample = min_n_windows  # use the minimum number of windows across all runs to ensure balanced sampling
            else:
                n_windows_to_sample = args.n_windows_per_run_for_metadata

        if n_windows_to_sample> 0:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(n_windows, n_windows_to_sample, replace=False)
            emb = emb[indices]
        n_windows = len(emb)

        X_plot.append(emb)   
        y_plot.append(np.array([meta_value] * n_windows))  # for plotting, repeat the meta value for each window
        X.append(emb.flatten())  # flatten the windowed embeddings for modeling                   
        y.append(meta_value)        
        groups.append(animal_id)     

    X_plot = np.concatenate(X_plot, axis=0)   # (n_total_windows, emb_dim)
    X = np.array(X)                            # (n_runs, n_windows * emb_dim)
    y_plot = np.concatenate(y_plot, axis=0)   # (n_total_windows,)
    y = np.array(y)                            # (n_runs,)
    groups = np.array(groups)                  # (n_runs,)
    print(f"Preprocessed metadata for {meta_var}: X shape {X.shape}, y shape {y.shape}, groups shape {groups.shape}")
    print(f"Plotting metadata for {meta_var}: X_plot shape {X_plot.shape}, y_plot shape {y_plot.shape}")

    if meta_var == "strain":
        valid_strains = metadata["strain"].value_counts()[metadata["strain"].value_counts() >= args.min_vids_per_strain].index.tolist()
        valid_strains = [s for s in valid_strains if s != "B6CAST-129SPWK-F2"]
        valid_indices = [i for i, label in enumerate(y) if label in valid_strains]
        X = X[valid_indices]           # was missing
        X_plot = X_plot[valid_indices] # note: index mismatch if n_windows != 1, see below
        y = y[valid_indices]
        y_plot = y_plot[valid_indices] # note: index mismatch if n_windows != 1, see below
        groups = groups[valid_indices]

    # if X is too large, we need need to apply PCA before modeling
    if X.shape[1] > 50:
        print(f"Applying PCA to reduce dimensionality of X from {X.shape[1]} to 50 for metadata classification.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50, random_state=args.seed)
        X = pca.fit_transform(X)
        print(f"Total Explained variance ratio of PCA components for metadata classification: {pca.explained_variance_ratio_.sum():.4f}")
        # Note: we do not apply PCA to X_plot since it's only used for visualization and we want to keep the original embedding space for that.

    return X, y, groups, X_plot, y_plot
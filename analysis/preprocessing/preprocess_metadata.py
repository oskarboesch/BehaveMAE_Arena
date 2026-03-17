from analysis.utils.window_and_aggregate import window_and_aggregate
import numpy as np

def preprocess_metadata(args, embeddings_dict, metadata, meta_var, token_shape, window_map):
    if window_map is not None:
        # some window has been applied to raw data during manifold process
        down_factor = args.window_size_manifold * token_shape[0]
    else :
        down_factor = token_shape[0]
    
    layer_window = max(1, args.window_size_metadata // down_factor)
    layer_stride = max(1, args.window_stride_metadata // down_factor)

    embeddings_windowed, _ = window_and_aggregate(
        embeddings_dict,
        window_size=layer_window,
        stride=layer_stride,
        method=args.agg_method
    )

    X = []
    X_plot = []  # for plotting only, not used in modeling
    y = []
    y_plot = []  # for plotting only, not used in modeling
    groups = []

    for run_id, emb in embeddings_windowed.items():  # iterate over windowed embeddings
        run_metadata = metadata[metadata["run_id"] == run_id]
        if len(run_metadata) == 0:
            print(f"No metadata found for run_id {run_id}, skipping.")
            continue
        elif len(run_metadata) > 1:
            print(f"Multiple metadata entries found for run_id {run_id}, using the first one.")

        meta_value = run_metadata.iloc[0][meta_var]
        animal_id = run_metadata.iloc[0]["animal_id"]
        n_windows = len(emb)
        # sample randomly args.n_windows windows 
        if n_windows < args.n_windows_per_run_for_metadata:
            raise ValueError(f"Not enough windows ({n_windows}) for run_id {run_id} to sample {args.n_windows_per_run_for_metadata} windows for metadata classification. Consider reducing the number of windows per run or adjusting the window size/stride.")
        indices = np.random.choice(n_windows, args.n_windows_per_run_for_metadata, replace=False)
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

    if meta_var == "strain":
        valid_strains = metadata["strain"].value_counts()[metadata["strain"].value_counts() >= args.min_vids_per_strain].index.tolist()
        valid_strains = [s for s in valid_strains if s != "B6CAST-129SPWK-F2"]
        valid_indices = [i for i, label in enumerate(y) if label in valid_strains]
        X = X[valid_indices]           # was missing
        X_plot = X_plot[valid_indices] # note: index mismatch if n_windows != 1, see below
        y = y[valid_indices]
        y_plot = y_plot[valid_indices] # note: index mismatch if n_windows != 1, see below
        groups = groups[valid_indices]

    return X, y, groups, X_plot, y_plot
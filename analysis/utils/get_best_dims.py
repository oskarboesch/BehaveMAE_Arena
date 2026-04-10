
import numpy as np
def get_best_dims(result, n_plot_features=None):
    raw_coefs = result["coefs"]

    # Check for None BEFORE wrapping in np.array
    if raw_coefs is None:
        return np.array([0, 1]) if n_plot_features is None or n_plot_features > 1 else np.array([0])

    coefs = np.array(raw_coefs)

    # if coefs are all zeros, return default dims
    if np.all(coefs == 0):
        return np.array([0, 1]) if n_plot_features is None or n_plot_features > 1 else np.array([0])

    # Convert to a 1D importance vector across modeled features.
    if coefs.ndim == 2:
        coef_importance = np.linalg.norm(coefs, axis=0)  # multi-class
    else:
        coef_importance = np.abs(coefs)  # single output

    # When training uses flattened windows (n_windows * emb_dim) but plotting
    # uses per-window embeddings (emb_dim), fold importances back to emb_dim.
    if n_plot_features is not None and n_plot_features > 0:
        if coef_importance.shape[0] == n_plot_features:
            pass
        elif coef_importance.shape[0] % n_plot_features == 0:
            coef_importance = coef_importance.reshape(-1, n_plot_features).sum(axis=0)
        else:
            # Fallback: keep only dimensions that exist in the plotted space.
            coef_importance = coef_importance[:n_plot_features]

    if coef_importance.shape[0] < 2:
        return np.array([0, 0])

    dims = np.argsort(coef_importance)[-2:]
    # return dims
    return (0,1) # for now let's only plot the first two dimensions, since the importance-based dims are not always stable across runs
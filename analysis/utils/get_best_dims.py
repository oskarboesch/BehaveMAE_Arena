
import numpy as np
def get_best_dims(result, n_plot_features=None):
    coefs = np.array(result["coefs"])

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
    return dims
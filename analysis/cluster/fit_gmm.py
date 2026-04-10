from  pyro import distributions as dist
import torch
from sklearn.mixture import GaussianMixture
import numpy as np

def _fit_gmm_pyro(
    X,
    n_components,
    max_iter=100,
    tol=1e-3,
    reg_covar=1e-6,
    random_state=42,
):
    """Fit a diagonal-covariance GMM with EM using Pyro distributions."""

    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN/Inf values before Pyro GMM fit")
    n_samples, n_features = X.shape

    if n_components < 2:
        raise ValueError("n_components must be >= 2 for GMM clustering")
    if n_components > n_samples:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed n_samples ({n_samples})"
        )

    rng = np.random.default_rng(random_state)
    init_ids = rng.choice(n_samples, size=n_components, replace=False)

    x_t = np.asarray(X)
    x_t_torch = torch.as_tensor(x_t, dtype=torch.float64)
    means = x_t[init_ids].copy()
    base_var = np.var(x_t, axis=0, ddof=0) + reg_covar
    variances = np.tile(base_var[None, :], (n_components, 1))
    weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)

    prev_ll = None
    eps = 1e-12

    for _ in range(max_iter):
        # E-step: compute responsibilities with Pyro log-probabilities.
        log_probs = np.empty((n_samples, n_components), dtype=np.float64)
        for k in range(n_components):
            scale_k = np.sqrt(np.maximum(variances[k], reg_covar))
            loc_k_torch = torch.as_tensor(means[k], dtype=torch.float64)
            scale_k_torch = torch.as_tensor(scale_k, dtype=torch.float64)
            comp = dist.Independent(
                dist.Normal(
                    loc=loc_k_torch,
                    scale=scale_k_torch,
                ),
                1,
            )
            log_probs[:, k] = comp.log_prob(x_t_torch).detach().cpu().numpy() + np.log(weights[k] + eps)

        if not np.isfinite(log_probs).all():
            raise FloatingPointError("Pyro GMM produced non-finite log-probabilities")

        max_log = np.max(log_probs, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=1, keepdims=True) + eps)
        resp = np.exp(log_probs - log_sum_exp)

        if not np.isfinite(resp).all():
            raise FloatingPointError("Pyro GMM produced non-finite responsibilities")

        ll = float(np.sum(log_sum_exp))

        # M-step
        Nk = np.sum(resp, axis=0) + eps
        weights = Nk / n_samples
        means = (resp.T @ x_t) / Nk[:, None]

        if not np.isfinite(means).all():
            raise FloatingPointError("Pyro GMM produced non-finite component means")

        for k in range(n_components):
            diff = x_t - means[k]
            variances[k] = (resp[:, k][:, None] * (diff ** 2)).sum(axis=0) / Nk[k]
            variances[k] = np.maximum(variances[k], reg_covar)

        if not np.isfinite(variances).all():
            raise FloatingPointError("Pyro GMM produced non-finite variances")

        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    labels = np.argmax(resp, axis=1)
    return labels, {"weights": weights, "means": means, "variances": variances, "log_likelihood": prev_ll}

def _predict_gmm_pyro(X, gmm_params):
    """Predict cluster labels for X using fitted Pyro GMM parameters."""
    X = np.asarray(X, dtype=np.float64)
    means = gmm_params["means"]
    variances = gmm_params["variances"]
    weights = gmm_params["weights"]
    n_components = len(weights)
    eps = 1e-12

    x_t_torch = torch.as_tensor(X, dtype=torch.float64)
    log_probs = np.empty((X.shape[0], n_components), dtype=np.float64)

    for k in range(n_components):
        scale_k = torch.as_tensor(np.sqrt(np.maximum(variances[k], 1e-6)), dtype=torch.float64)
        loc_k = torch.as_tensor(means[k], dtype=torch.float64)
        comp = dist.Independent(dist.Normal(loc=loc_k, scale=scale_k), 1)
        log_probs[:, k] = comp.log_prob(x_t_torch).detach().cpu().numpy() + np.log(weights[k] + eps)

    return np.argmax(log_probs, axis=1)

def _sanitize_embeddings_for_gmm(X):
    """Return a finite float64 matrix for GMM while preserving sample count."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for GMM, got shape {X.shape}")

    # Replace NaN/Inf per feature with finite column mean, or 0.0 if column is fully non-finite.
    finite_mask = np.isfinite(X)
    if finite_mask.all():
        return X

    X_clean = X.copy()
    n_replaced = int((~finite_mask).sum())

    for j in range(X_clean.shape[1]):
        col = X_clean[:, j]
        good = np.isfinite(col)
        if good.any():
            fill_value = float(np.mean(col[good]))
        else:
            fill_value = 0.0
        col[~good] = fill_value
        X_clean[:, j] = col

    print(f"Warning: replaced {n_replaced} non-finite values in embeddings before GMM.")
    return X_clean


def _fit_gmm_sklearn_fallback(X, n_components, reg_covar=1e-6, random_state=42, max_iter=100):
    """Fallback GMM using sklearn when Pyro EM is numerically unstable."""
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=reg_covar,
        random_state=random_state,
        max_iter=max_iter,
    )
    labels = gm.fit_predict(X)
    # sklearn diag covariance gives shape [k, n_features]
    variances = np.asarray(gm.covariances_)
    if variances.ndim == 3:
        variances = np.diagonal(variances, axis1=1, axis2=2)

    return labels, {
        "weights": np.asarray(gm.weights_),
        "means": np.asarray(gm.means_),
        "variances": variances,
        "log_likelihood": float(gm.lower_bound_),
    }

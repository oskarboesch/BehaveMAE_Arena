import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyro import distributions as dist
import seaborn as sns

from .plot_hdbscan_silhouette import _safe_silhouette_score


def _fit_gmm_pyro_labels(
	X,
	n_components,
	max_iter=100,
	tol=1e-3,
	reg_covar=1e-6,
	random_state=42,
):
	"""Fit a diagonal-covariance GMM with Pyro and return hard labels."""
	X = np.asarray(X, dtype=np.float64)
	n_samples, _ = X.shape

	if n_components < 2:
		raise ValueError("n_components must be >= 2")
	if n_components > n_samples:
		raise ValueError(
			f"n_components ({n_components}) cannot exceed n_samples ({n_samples})"
		)

	rng = np.random.default_rng(random_state)
	init_ids = rng.choice(n_samples, size=n_components, replace=False)

	x_np = np.asarray(X)
	x_torch = torch.as_tensor(x_np, dtype=torch.float64)

	means = x_np[init_ids].copy()
	base_var = np.var(x_np, axis=0, ddof=0) + reg_covar
	variances = np.tile(base_var[None, :], (n_components, 1))
	weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)

	eps = 1e-12
	prev_ll = None

	for _ in range(max_iter):
		log_probs = np.empty((n_samples, n_components), dtype=np.float64)

		for k in range(n_components):
			scale_k = np.sqrt(np.maximum(variances[k], reg_covar))
			comp = dist.Independent(
				dist.Normal(
					loc=torch.as_tensor(means[k], dtype=torch.float64),
					scale=torch.as_tensor(scale_k, dtype=torch.float64),
				),
				1,
			)
			log_probs[:, k] = (
				comp.log_prob(x_torch).detach().cpu().numpy() + np.log(weights[k] + eps)
			)

		max_log = np.max(log_probs, axis=1, keepdims=True)
		log_sum_exp = max_log + np.log(
			np.sum(np.exp(log_probs - max_log), axis=1, keepdims=True) + eps
		)
		resp = np.exp(log_probs - log_sum_exp)

		ll = float(np.sum(log_sum_exp))

		Nk = np.sum(resp, axis=0) + eps
		weights = Nk / n_samples
		means = (resp.T @ x_np) / Nk[:, None]

		for k in range(n_components):
			diff = x_np - means[k]
			variances[k] = (resp[:, k][:, None] * (diff ** 2)).sum(axis=0) / Nk[k]
			variances[k] = np.maximum(variances[k], reg_covar)

		if prev_ll is not None and abs(ll - prev_ll) < tol:
			break
		prev_ll = ll

	return np.argmax(resp, axis=1)


def plot_gmm_silhouettes(
	layers_embeddings,
	k_range=(2, 11),
	max_iter=100,
	tol=1e-3,
	reg_covar=1e-6,
	random_state=42,
	savepath=None,
):
	"""Plot silhouette score vs number of components for per-layer Pyro GMM."""
	K_range = range(k_range[0], k_range[1])
	silhouette_scores = {i: [] for i in range(len(layers_embeddings))}

	for k in tqdm(K_range, desc="GMM Silhouette"):
		for layer_idx, emb in enumerate(layers_embeddings):
			emb = np.asarray(emb)
			if emb.shape[0] < k:
				print(
					f"Warning: not enough samples ({emb.shape[0]}) for GMM with k={k} at layer {layer_idx}."
				)
				silhouette_scores[layer_idx].append(0.0)
				continue

			try:
				labels = _fit_gmm_pyro_labels(
					emb,
					n_components=k,
					max_iter=max_iter,
					tol=tol,
					reg_covar=reg_covar,
					random_state=random_state,
				)
				sil = _safe_silhouette_score(emb, labels=labels, eps=f"gmm_k={k}", layer_idx=layer_idx)
			except Exception as e:
				print(
					f"Warning: GMM silhouette failed for layer {layer_idx}, k={k}: {e}. Using 0."
				)
				sil = 0.0

			silhouette_scores[layer_idx].append(float(sil))

	plt.figure(figsize=(10, 6))
	for i in range(len(layers_embeddings)):
		sns.lineplot(
			x=list(K_range),
			y=silhouette_scores[i],
			label=f"Layer {i}",
			marker='o',
			markersize=6,
			linewidth=1.8,
		)

	plt.xlabel("Number of components (k)")
	plt.xticks(list(K_range))
	plt.ylabel("Silhouette Score")
	plt.title("Silhouette Scores for Each Layer (GMM-Pyro)")
	plt.grid(alpha=0.25, linestyle='--', linewidth=0.6)
	plt.legend()
	plt.tight_layout()
	if savepath is not None:
		plt.savefig(savepath)
	plt.show()

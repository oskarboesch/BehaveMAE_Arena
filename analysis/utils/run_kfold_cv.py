from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import accuracy_score
import numpy as np
from .get_stats_report import get_stats_report

MAX_STATS_FEATURES_CLASSIFICATION = 1000
MAX_STATS_FEATURES_REGRESSION = 2000
MAX_HESSIAN_MB = 128


def _should_skip_stats_report(X, is_classification):
    n_samples, n_features = X.shape
    max_features = (
        MAX_STATS_FEATURES_CLASSIFICATION if is_classification else MAX_STATS_FEATURES_REGRESSION
    )

    if n_features > max_features:
        return True, f"too many features ({n_features} > {max_features})"

    # Hessian-like objects scale quadratically in feature count.
    approx_hessian_mb = (n_features * n_features * 8) / (1024 * 1024)
    if approx_hessian_mb > MAX_HESSIAN_MB:
        return True, f"estimated Hessian size too large (~{approx_hessian_mb:.1f} MB > {MAX_HESSIAN_MB} MB)"

    if is_classification and n_features >= n_samples:
        return True, f"high-dimensional regime (n_features={n_features} >= n_samples={n_samples})"

    return False, None

def run_kfold_cv(results, var, model, dummy_model, X, y, groups=None, is_classification=True, seed=42):
    if is_classification:
        if var == "syllable":
            cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed) # impossible to stratify with syllable labels, but we want to keep samples from the same animal together in the same fold to avoid data leakage
        else:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        score_func = accuracy_score
        scores = []
        train_scores = []
        dummy_model_scores = []
        print(f"Running StratifiedGroupKFold CV for variable '{var}' with {cv.get_n_splits(groups=groups)} splits.")
        print(f"X shape : {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution in y: {dict(zip(unique, counts))}")
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
            model.fit(X[train_idx], y[train_idx])
            dummy_model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_dummy_pred = dummy_model.predict(X[test_idx])
            if is_classification:
                score = score_func(y[test_idx], y_pred)
                dummy_score = score_func(y[test_idx], y_dummy_pred)
            else:
                score = score_func(y[test_idx], y_pred)
                dummy_score = score_func(y[test_idx], y_dummy_pred)

            scores.append(score)
            train_scores.append(model.score(X[train_idx], y[train_idx]))
            dummy_model_scores.append(dummy_score)
    else:
        # only stats are relevant for kinematic analysis
        scores = [0]
        train_scores = [0]
        dummy_model_scores = [0]

    # Statsmodels can fail on ill-conditioned folds/data (e.g. perfect separation,
    # non-invertible Hessian). Keep pipeline alive and store CV metrics anyway.
    result = None
    stats_error = None
    skip_stats, skip_reason = _should_skip_stats_report(X, is_classification)
    if skip_stats:
        stats_error = f"skipped stats report: {skip_reason}"
    else:
        try:
            result = get_stats_report(X, y, is_classification=is_classification)
        except Exception as exc:
            stats_error = str(exc)

    if result is not None:
        try:
            r2_value = result.prsquared if is_classification else result.rsquared
        except Exception:
            r2_value = None

        try:
            coefs = np.asarray(result.params).reshape(-1).tolist()[:10]
        except Exception:
            coefs = None

        try:
            pvalues = np.asarray(result.pvalues).reshape(-1).tolist()[:10]
        except Exception as exc:
            pvalues = None
            if stats_error is None:
                stats_error = str(exc)
    else:
        r2_value = None
        coefs = None
        pvalues = None

    if stats_error is not None:
        print(f"Warning: stats report unavailable for '{var}': {stats_error}")

    results[var] = {
        "train_accuracy": np.mean(train_scores),
        "accuracy": np.mean(scores),
        "std": np.std(scores),
        f"dummy_accuracy": np.mean(dummy_model_scores),
        "dummy_std": np.std(dummy_model_scores), 
        "r2": r2_value,
        "coefs": coefs,
        "pvalues": pvalues,
        "stats_error": stats_error,
    }
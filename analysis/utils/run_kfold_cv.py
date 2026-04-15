from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from .get_stats_report import get_stats_report
from sklearn.base import clone

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

def run_kfold_cv(results, var, model, dummy_model, X, y, groups=None, is_classification=True, seed=42, run_ids=None):
    if is_classification:
        if var == "syllable":
            cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed) # impossible to stratify with syllable labels, but we want to keep samples from the same animal together in the same fold to avoid data leakage
        else:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        
        scores = []
        train_scores = []
        dummy_model_scores = []
        f1_scores = []
        f1_dummy_scores = []
        f1_seq_scores = []
        f1_seq_dummy_scores = []
        print(f"Running StratifiedGroupKFold CV for variable '{var}' with {cv.get_n_splits(groups=groups)} splits.")
        print(f"X shape : {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution in y: {dict(zip(unique, counts))}")
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
            model.fit(X[train_idx], y[train_idx])
            dummy_model.fit(X[train_idx], y[train_idx])

            # Window-level predictions
            y_pred = model.predict(X[test_idx])
            y_dummy_pred = dummy_model.predict(X[test_idx])
            
            score = accuracy_score(y[test_idx], y_pred)
            dummy_score = accuracy_score(y[test_idx], y_dummy_pred)
            
            # F1 score at window level
            f1_window = f1_score(y[test_idx], y_pred, average='macro' if len(unique) > 2 else 'binary', pos_label=unique[1] if len(unique) == 2 else 1)
            f1_dummy_window = f1_score(y[test_idx], y_dummy_pred, average='macro' if len(unique) > 2 else 'binary', pos_label=unique[1] if len(unique) == 2 else 1)
            f1_scores.append(f1_window)
            f1_dummy_scores.append(f1_dummy_window)

            # Sequence-level predictions if run_ids are provided
            if run_ids is not None:
                test_run_ids = run_ids[test_idx]
                
                # --- Real model: aggregate window probas per run ---
                if hasattr(model, "predict_proba"):
                    y_probas = model.predict_proba(X[test_idx])
                    df_preds = pd.DataFrame(y_probas, columns=model.classes_)
                    df_preds['run_id'] = test_run_ids
                    df_preds['y_true'] = y[test_idx]
                    
                    grouped = df_preds.groupby('run_id')
                    mean_probas = grouped[model.classes_].mean()
                    y_true_seq = grouped['y_true'].first().values
                    y_pred_seq = model.classes_[np.argmax(mean_probas.values, axis=1)]
                    
                    f1_seq = f1_score(y_true_seq, y_pred_seq, average='macro' if len(unique) > 2 else 'binary', pos_label=unique[1] if len(unique) == 2 else 1)
                    f1_seq_scores.append(f1_seq)

                    # --- Dummy: operate directly at sequence level ---
                    # Build sequence-level labels (one label per run_id)
                    df_runs = pd.DataFrame({'run_id': test_run_ids, 'y_true': y[test_idx]})
                    y_true_seq_for_dummy = df_runs.groupby('run_id')['y_true'].first().values

                    # Also need sequence-level train labels
                    train_run_ids = run_ids[train_idx]
                    df_train_runs = pd.DataFrame({'run_id': train_run_ids, 'y_true': y[train_idx]})
                    y_train_seq = df_train_runs.groupby('run_id')['y_true'].first().values

                    # Fit and predict dummy at sequence level
                    dummy_seq = clone(dummy_model)  # fresh copy, don't reuse window-level fit
                    dummy_seq.fit(np.zeros((len(y_train_seq), 1)), y_train_seq)
                    y_dummy_pred_seq = dummy_seq.predict(np.zeros((len(y_true_seq_for_dummy), 1)))

                    f1_dummy_seq = f1_score(y_true_seq_for_dummy, y_dummy_pred_seq, average='macro' if len(unique) > 2 else 'binary', pos_label=unique[1] if len(unique) == 2 else 1)
                    f1_seq_dummy_scores.append(f1_dummy_seq)

            scores.append(score)
            train_scores.append(model.score(X[train_idx], y[train_idx]))
            dummy_model_scores.append(dummy_score)
    else:
        # only stats are relevant for kinematic analysis
        scores = [0]
        train_scores = [0]
        dummy_model_scores = [0]
        f1_scores = []
        f1_dummy_scores = []
        f1_seq_scores = []
        f1_seq_dummy_scores = []

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

    res_dict = {
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
    if is_classification:
        res_dict["f1_window"] = np.mean(f1_scores) if f1_scores else None
        res_dict["f1_seq"] = np.mean(f1_seq_scores) if f1_seq_scores else None
        res_dict["f1_window_std"] = np.std(f1_scores) if f1_scores else None
        res_dict["f1_seq_std"] = np.std(f1_seq_scores) if f1_seq_scores else None
        res_dict["f1_dummy_window"] = np.mean(f1_dummy_scores) if f1_dummy_scores else None
        res_dict["f1_dummy_seq"] = np.mean(f1_seq_dummy_scores) if f1_seq_dummy_scores else None

    results[var] = res_dict
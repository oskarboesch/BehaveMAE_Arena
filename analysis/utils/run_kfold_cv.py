from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import accuracy_score
import numpy as np
from .get_stats_report import get_stats_report

def run_kfold_cv(results, var, model, dummy_model, X, y, groups=None, is_classification=True):
    if is_classification:
        if var == "syllable":
            cv = GroupKFold(n_splits=5, shuffle=True, random_state=42) # impossible to stratify with syllable labels, but we want to keep samples from the same animal together in the same fold to avoid data leakage
        else:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        score_func = accuracy_score
        scores = []
        train_scores = []
        dummy_model_scores = []
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

    # statsmodels for the statistical report (fit scaler on train only)
    result = get_stats_report(X, y, is_classification=is_classification)

    results[var] = {
        "train_accuracy": np.mean(train_scores),
        "accuracy": np.mean(scores),
        "std": np.std(scores),
        f"dummy_accuracy": np.mean(dummy_model_scores),
        "dummy_std": np.std(dummy_model_scores), 
        "r2": result.prsquared if is_classification else result.rsquared,
        "coefs": result.params.tolist()[:10],
        "pvalues": result.pvalues.tolist()[:10]
    }
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm

def get_stats_report(X, y, is_classification=True):
    X_scaled = StandardScaler().fit_transform(X)

    if is_classification:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        n_classes = len(le.classes_)
        if n_classes == 2:
            model_sm = sm.Logit(y_enc, X_scaled)
        else:
            model_sm = sm.MNLogit(y_enc, X_scaled)

        result = model_sm.fit(method="lbfgs", maxiter=2000, disp=False)
    else:
        model_sm = sm.OLS(y, X_scaled)
        result = model_sm.fit(method="pinv")

    return result
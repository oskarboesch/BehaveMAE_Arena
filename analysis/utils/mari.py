import numpy as np

def mari(y1, y2):
    """Compute MARI (Modified Adjusted Rand Index) for two categorical label vectors.
 
    Based on Sundqvist, Chiquet & Rigaill (2020): "Adjusting the adjusted
    Rand Index - A multinomial story".
 
    Works for any discrete label vectors (binary, multi-class, cluster ids, etc.).
 
    MARI = theta_hat - theta_hat_0  (Eq. 12)
 
      theta_hat   = sum_{k,l} C(n_kl, 2) / C(n, 2)           [Eq. 4]
                    Fraction of pairs assigned to the same cluster
                    in both classifications.
 
      theta_hat_0 = sum_Q / (6 * C(n, 4))                     [Eq. 12]
                    Multinomial chance-correction computed from
                    the contingency table via Lemma 7.
 
    Returns a float in roughly [-0.5, 1.0]:
        1.0  → perfect agreement
        ~0   → agreement at chance level (independent clusterings)
        <0   → systematic anti-correlation
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    n = len(y1)
    if n < 4:
        return 0.0
 
    def c2(x):
        return x * (x - 1) // 2
 
    def c4(x):
        return x * (x - 1) * (x - 2) * (x - 3) // 24
 
    cn2 = c2(n)
    if cn2 == 0:
        return 0.0
 
    # --- Build sparse contingency table via sorted unique pairs ---
    # Each (k, l) pair maps to a count n_kl.
    keys, counts = np.unique(np.stack([y1, y2], axis=1), axis=0, return_counts=True)
    # nkl_vals: the non-zero entries of the contingency table
    nkl_vals = counts.astype(np.int64)
 
    # Row/column marginals from the sparse table
    row_labels, row_inv = np.unique(keys[:, 0], return_inverse=True)
    col_labels, col_inv = np.unique(keys[:, 1], return_inverse=True)
    nk = np.bincount(row_inv, weights=nkl_vals).astype(np.int64)  # n_k.
    nl = np.bincount(col_inv, weights=nkl_vals).astype(np.int64)  # n_.l
 
    # --- theta_hat (Eq. 4) ---
    sum_c2_nkl = int(np.sum(nkl_vals * (nkl_vals - 1) // 2))
    theta_hat = sum_c2_nkl / cn2
 
    # --- theta_hat_0 via Lemma 7 ---
    sum_nk2     = int(np.sum(nk ** 2))
    sum_nl2     = int(np.sum(nl ** 2))
    sum_nkl2    = int(np.sum(nkl_vals ** 2))
    sum_nk2_nl2 = sum_nk2 * sum_nl2
    sum_nk_c2   = int(np.sum(nk * (nk - 1) // 2))
    sum_nl_c2   = int(np.sum(nl * (nl - 1) // 2))
 
    # cross term: sum_{k,l} n_k. * n_kl * n_.l  (Lemma 6)
    sum_cross = int(np.sum(nk[row_inv] * nkl_vals * nl[col_inv]))
 
    # Lemma 6: sum_T
    sum_T = (2 * n
             + sum_cross
             - sum_nkl2
             - sum_nk2
             - sum_nl2)
 
    # Lemma 7: sum_Q
    sum_Q = (sum_nk2_nl2
             - 4 * sum_c2_nkl
             + 4 * sum_T
             + 2 * n * (sum_nk_c2 + sum_nl_c2)
             + n ** 2) / 4
 
    cn4_6 = 6 * c4(n)
    if cn4_6 == 0:
        return 0.0
 
    theta_hat_0 = sum_Q / cn4_6
 
    return float(theta_hat - theta_hat_0)

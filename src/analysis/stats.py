"""Statistical helpers shared by the biomarker and genomic analyses."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def numeric_tests(df: pd.DataFrame, cols, y: pd.Series) -> pd.DataFrame:
    """
    Compare the positive class (y=1) with the negative class (y=0) for each
    numeric column.

    Returns medians and IQRs per class, the Mann-Whitney U p-value, BH q-value,
    rank-biserial correlation (effect size, -1..1, positive = higher in the
    positive class) and the single-feature ROC AUC (direction-free, >= 0.5).
    """
    rows = []
    for c in cols:
        a = df.loc[y == 1, c].dropna()
        b = df.loc[y == 0, c].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 2 * u / (len(a) * len(b)) - 1
        mask = df[c].notna()
        auc = roc_auc_score(y[mask], df.loc[mask, c])
        rows.append({
            "feature": c,
            "negative_class_median": b.median(),
            "negative_class_iqr": f"{b.quantile(.25):.3g}-{b.quantile(.75):.3g}",
            "positive_class_median": a.median(),
            "positive_class_iqr": f"{a.quantile(.25):.3g}-{a.quantile(.75):.3g}",
            "mannwhitney_p": p,
            "rank_biserial": rbc,
            "univariate_auc": max(auc, 1 - auc),
        })
    out = pd.DataFrame(rows)
    out["q_value"] = bh_fdr(out["mannwhitney_p"])
    return out.sort_values("q_value").reset_index(drop=True)


def categorical_tests(df: pd.DataFrame, cols, y: pd.Series) -> pd.DataFrame:
    """Chi-square test of independence + Cramer's V for each categorical column."""
    rows = []
    for c in cols:
        tab = pd.crosstab(df[c], y)
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        n = tab.values.sum()
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        rows.append({"feature": c, "chi2": chi2, "dof": dof,
                     "chi2_p": p, "cramers_v": v})
    out = pd.DataFrame(rows)
    out["q_value"] = bh_fdr(out["chi2_p"])
    return out.sort_values("q_value").reset_index(drop=True)


def level_enrichment(df: pd.DataFrame, col: str, y: pd.Series) -> pd.DataFrame:
    """
    Fisher's exact test for each level of a categorical column
    (level vs all other levels), giving an odds ratio of the positive class.
    A 0.5 continuity correction keeps the log-OR finite for sparse cells.
    """
    rows = []
    s = df[col]
    keep = s.notna()
    s, yy = s[keep], y[keep]
    for lvl in sorted(s.unique()):
        in_l = s == lvl
        a = int(((yy == 1) & in_l).sum()); b = int(((yy == 0) & in_l).sum())
        c = int(((yy == 1) & ~in_l).sum()); d = int(((yy == 0) & ~in_l).sum())
        _, p = stats.fisher_exact([[a, b], [c, d]])
        a2, b2, c2, d2 = a + .5, b + .5, c + .5, d + .5
        lor = np.log(a2 * d2 / (b2 * c2))
        se = np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
        rows.append({"feature": col, "level": lvl, "n": a + b,
                     "positive_rate": a / max(a + b, 1),
                     "odds_ratio": np.exp(lor),
                     "or_ci_low": np.exp(lor - 1.96 * se),
                     "or_ci_high": np.exp(lor + 1.96 * se),
                     "fisher_p": p})
    return pd.DataFrame(rows)

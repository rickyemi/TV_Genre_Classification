"""
Stage 3: preprocessing and feature selection (fitted on the 80% training split).

Missing values are resolved in two layers:
1. Data stage (deterministic recovery): D / F gaps come from their complete
   copy I; missing J is reconstructed exactly from B, C, I, L, M, N.
2. Model pipeline (safety net for future feeds): median imputation. The
   figure here shows why exact recovery beats statistical imputation for J.

Model preprocessing pipeline: IQR winsorisation (Tukey fences learned on
train) -> median imputation -> z-score (StandardScaler).

Feature selection on the training split. Rules:
  a. association with genre: Kruskal-Wallis, Benjamini-Hochberg q < 0.05
  b. information: mutual information >= 0.01 nats
  c. no exact linear redundancy: while any VIF > 1000, drop the feature with
     the lowest mutual information among those with VIF > 1000
  d. latent basis: B, C, I, J, L, M, N span only 5 dimensions (rank check);
     C, I, L, M, N are 5 measured, complete, linearly independent columns that
     reproduce B and J exactly (R^2 = 1), so they carry all of that information
     without redundancy
Five strategies are compared with repeated 5-fold CV on the training split
(reference models: SVM-RBF, XGBoost and a class-conditional Gaussian
mixture; score = the worst of the five weighted target metrics):
  S1 a+b+c   S2 a+b   S3 a only   S4 all 12 cleaned features   S5 latent basis
and the strategy with the best reference score is used. The holdout is never
touched.

Run:  python -m src.features.build_features
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src.analysis.stats import bh_fdr
from src.data.make_dataset import load
from src.utils import get_logger, load_json, save_json
from src.visualization.visualize import DIV_CMAP, save_fig

log = get_logger(__name__)


class IQRCapper(TransformerMixin, BaseEstimator):
    """Winsorise each column at Tukey fences learned on the training data."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        self.lower_, self.upper_ = q1 - self.factor * (q3 - q1), q3 + self.factor * (q3 - q1)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor() -> Pipeline:
    return Pipeline([("outlier_cap", IQRCapper(1.5)), ("impute", SimpleImputer(strategy="median")),
                     ("zscore", StandardScaler())])


def vif(df: pd.DataFrame) -> pd.Series:
    out = {}
    for c in df.columns:
        others = [o for o in df.columns if o != c]
        r2 = LinearRegression().fit(df[others], df[c]).score(df[others], df[c]) if others else 0.0
        out[c] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return pd.Series(out)


def select_features(train: pd.DataFrame) -> dict:
    X, y = train[cfg.CLEAN_FEATURES], train[cfg.TARGET]
    rows = []
    mi = mutual_info_classif(X, y, random_state=cfg.RANDOM_STATE)
    for c, m in zip(cfg.CLEAN_FEATURES, mi):
        rows.append({"feature": c, "kruskal_p": stats.kruskal(*[X.loc[y == g, c] for g in cfg.CLASSES]).pvalue,
                     "mutual_info": m})
    t = pd.DataFrame(rows)
    t["q_value"] = bh_fdr(t.kruskal_p.values)
    t["pass_fdr"], t["pass_mi"] = t.q_value < cfg.FDR_ALPHA, t.mutual_info >= cfg.MIN_MUTUAL_INFO
    cand = t.loc[t.pass_fdr & t.pass_mi, "feature"].tolist()
    mi_s = t.set_index("feature").mutual_info
    vif_before = vif(X[cand])
    dropped_vif, keep = [], list(cand)
    while True:
        v = vif(X[keep])
        bad = v[v > cfg.VIF_MAX]
        if bad.empty:
            break
        loser = mi_s[bad.index].idxmin()
        dropped_vif.append(loser); keep.remove(loser)
    vif_after = vif(X[keep])
    t["selected"] = t.feature.isin(keep)
    t["reason"] = np.select([t.selected, ~t.pass_fdr, ~t.pass_mi, t.feature.isin(dropped_vif)],
                            ["kept", "dropped: FDR q >= 0.05", "dropped: MI < 0.01",
                             "dropped: exact linear redundancy (VIF)"], "dropped")
    t["vif_among_candidates"] = t.feature.map(vif_before)
    t["vif_final"] = t.feature.map(vif_after)
    basis = latent_basis_check(train)
    strategies = {"S1 FDR + MI + VIF": keep, "S2 FDR + MI": cand,
                  "S3 FDR only": t.loc[t.pass_fdr, "feature"].tolist(), "S4 all cleaned": list(cfg.CLEAN_FEATURES),
                  "S5 latent basis": list(cfg.LATENT_BASIS)}
    strategies = {k: [f for f in cfg.CLEAN_FEATURES if f in v] for k, v in strategies.items()}
    comp = compare_strategies(train, strategies)
    best = comp.groupby("strategy").score.max().idxmax()
    final = strategies[best]
    t["selected"] = t.feature.isin(final)
    t["in_S1"] = t.feature.isin(keep)
    t["reason"] = np.select([t.selected, ~t.pass_fdr, ~t.pass_mi, t.feature.isin(dropped_vif)],
                            ["kept", "dropped: FDR q >= 0.05", "dropped: MI < 0.01",
                             "dropped: exact linear redundancy (VIF)"], "dropped")
    return {"selected": final, "strategy": best, "strategies": strategies, "comparison": comp,
            "dropped_vif": dropped_vif, "latent_basis": basis, "table": t.sort_values("mutual_info", ascending=False)}


def latent_basis_check(train: pd.DataFrame) -> dict:
    """Rank of the linearly dependent block and how well the basis reproduces every block column."""
    block = cfg.J_SOURCES + ["J"]
    Z = train[block].values - train[block].values.mean(0)
    sv = np.linalg.svd(Z, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-9))
    B = train[cfg.LATENT_BASIS]
    r2 = {c: float(LinearRegression().fit(B, train[c]).score(B, train[c])) for c in block + ["A"]}
    log.info("Linear block %s has rank %d; basis %s reproduces: %s", block, rank, cfg.LATENT_BASIS,
             {k: round(v, 4) for k, v in r2.items()})
    return {"block": block, "rank": rank, "basis": list(cfg.LATENT_BASIS), "r2_from_basis": r2,
            "basis_rank": int(np.linalg.matrix_rank(B.values - B.values.mean(0)))}


def compare_strategies(train, strategies) -> pd.DataFrame:
    """Repeated stratified 5-fold CV (3 repeats) of three reference models per strategy."""
    import warnings
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from src.models.gmm_models import SemiSupervisedGMMClassifier
    from src.models.metrics import MIN_WEIGHTED_SCORER
    y = train[cfg.TARGET].map({c: i for i, c in enumerate(cfg.CLASSES)})
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=cfg.RANDOM_STATE)
    refs = {"SVM-RBF (C=10, balanced)": lambda: Pipeline([("prep", build_preprocessor()),
                                                          ("clf", SVC(C=10, class_weight="balanced"))]),
            "XGBoost (depth 3, 300 trees)": lambda: Pipeline([("prep", build_preprocessor()), (
                "clf", XGBClassifier(max_depth=3, n_estimators=300, learning_rate=.05, n_jobs=1, verbosity=0))]),
            "Gaussian mixture (2 per genre)": lambda: Pipeline([("impute", SimpleImputer(strategy="median")), (
                "clf", SemiSupervisedGMMClassifier(2, use_unlabelled=False, reg_covar=1e-3))])}
    rows = []
    for sname, feats in strategies.items():
        for rname, mk in refs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sc = cross_val_score(mk(), train[feats], y, cv=cv, scoring=MIN_WEIGHTED_SCORER)
            rows.append({"strategy": sname, "n_features": len(feats), "reference_model": rname,
                         "score": sc.mean(), "sd": sc.std(), "features": ",".join(feats)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def plot_missing_strategy(lab, spec):
    """For shows where J IS observed, hide it and compare recovery methods."""
    d = lab[lab.J_was_reconstructed == 0].reset_index(drop=True)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    hide = rng.random(len(d)) < .2
    truth = d.loc[hide, "J"].values
    exact = spec["j_intercept"] + sum(d.loc[hide, c] * w for c, w in spec["j_coef"].items())
    med = np.full(hide.sum(), d.loc[~hide, "J"].median())
    knn_in = d[cfg.CLEAN_FEATURES].copy(); knn_in.loc[hide, "J"] = np.nan
    knn = KNNImputer(n_neighbors=5).fit_transform(knn_in)[hide, cfg.CLEAN_FEATURES.index("J")]
    errs = {"Median": np.abs(med - truth), "KNN (k=5)": np.abs(knn - truth), "Exact formula": np.abs(exact - truth)}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    flow = pd.DataFrame({"Raw missing": [lab.J_was_reconstructed.sum(), 0],
                         "After cleaning": [0, 0]}, index=["J", "D/F"])
    raw = pd.read_csv(cfg.RAW_TRAIN)
    a1.bar(["D", "F", "J"], [raw.D.isna().sum(), raw.F.isna().sum(), raw.J.isna().sum()], color="#86b6ef",
           width=.55, label="Missing in raw training file")
    a1.bar(["D", "F", "J"], [0, 0, 0], color="#1c5cab", width=.55)
    for i, (v, how) in enumerate(zip([raw.D.isna().sum(), raw.F.isna().sum(), raw.J.isna().sum()],
                                     ["dropped (copy of I)", "dropped (copy of I)", "rebuilt exactly"])):
        a1.text(i, v + 1, f"{v}\n{how}", ha="center", fontsize=8)
    a1.set_ylim(0, 75); a1.set_ylabel("Missing values"); a1.grid(axis="x", visible=False)
    a1.set_title("How each gap is resolved (0 left after cleaning)")
    del flow
    names = list(errs)
    a2.bar(names, [errs[n].mean() for n in names], color=["#8a8985", "#eb6834", "#1baf7a"], width=.55)
    for i, n in enumerate(names):
        a2.text(i, errs[n].mean() + .02, f"{errs[n].mean():.3f}", ha="center", fontsize=9)
    a2.set_ylabel("Mean absolute error on hidden J"); a2.grid(axis="x", visible=False)
    a2.set_title(f"Imputation test: hide 20% of observed J (n={hide.sum()})")
    a3.scatter(truth, med, s=10, color="#8a8985", label="Median", alpha=.6)
    a3.scatter(truth, knn, s=10, color="#eb6834", label="KNN", alpha=.6)
    a3.scatter(truth, exact, s=10, color="#1baf7a", label="Exact formula")
    a3.plot([truth.min(), truth.max()], [truth.min(), truth.max()], color=cfg.NEUTRAL, ls="--", lw=1)
    a3.set_xlabel("True J"); a3.set_ylabel("Imputed J"); a3.legend(fontsize=7.5); a3.set_title("Imputed vs true J")
    fig.tight_layout()
    save_fig(fig, "12_missing_value_handling", "Median imputation stays in the pipeline as a safety net for future feeds")
    return pd.DataFrame({"method": names, "MAE": [errs[n].mean() for n in names]})


def outlier_table(train):
    X = train[cfg.CLEAN_FEATURES]
    cap = IQRCapper().fit(X)
    rows = []
    for i, c in enumerate(cfg.CLEAN_FEATURES):
        s = X[c]
        o = (s < cap.lower_[i]) | (s > cap.upper_[i])
        rows.append({"feature": c, "lower_fence": cap.lower_[i], "upper_fence": cap.upper_[i],
                     "n_outliers": int(o.sum()), "pct_outliers": 100 * o.mean(),
                     "n_abs_z_gt_3": int((np.abs((s - s.mean()) / s.std()) > 3).sum())})
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)


def plot_outliers(train, table):
    raw = pd.read_csv(cfg.RAW_TRAIN)
    X = train[cfg.CLEAN_FEATURES]
    Z = (X - X.mean()) / X.std()
    Zc = (pd.DataFrame(IQRCapper().fit_transform(X), columns=X.columns) - X.mean()) / X.std()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), gridspec_kw={"width_ratios": [.7, 1, 1, .8]})
    za = (raw.A - raw.A.mean()) / raw.A.std()
    axes[0].boxplot([za, Z["A"]], widths=.5, patch_artist=True,
                    flierprops=dict(marker="o", ms=3, mfc="#eb6834", mec="none"))
    axes[0].set_xticks([1, 2], ["raw A", "cleaned A"]); axes[0].set_ylabel("z-score")
    axes[0].set_title("Unit-error outliers in A", fontsize=10)
    for ax, data, title in [(axes[1], Z, "Cleaned features, before capping"), (axes[2], Zc, "After IQR capping")]:
        bp = ax.boxplot([data[c] for c in X.columns], vert=False, widths=.55, patch_artist=True,
                        flierprops=dict(marker="o", ms=2.5, mfc="#eb6834", mec="none", alpha=.6),
                        medianprops=dict(color=cfg.TEXT_PRIMARY))
        for p in bp["boxes"]:
            p.set_facecolor("#b7d3f6"); p.set_edgecolor("#3987e5")
        ax.set_yticks(range(1, len(X.columns) + 1), X.columns)
        for v in (-3, 3):
            ax.axvline(v, color=cfg.NEUTRAL, ls=":", lw=1)
        ax.set_xlabel("z-score"); ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False)
    t = table.sort_values("pct_outliers")
    axes[3].barh(t.feature, t.pct_outliers, color="#eb6834", height=.6)
    axes[3].set_xlabel("% beyond IQR fences"); axes[3].grid(axis="y", visible=False)
    axes[3].set_title("Remaining outliers (genuine tails)", fontsize=10)
    fig.suptitle("Outlier analysis: the A unit error is fixed in cleaning; remaining tails are winsorised, not dropped",
                 x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "13_outlier_analysis")


def plot_zscore(train, selected):
    show = selected[:4]
    prep = build_preprocessor().fit(train[selected])
    Xt = pd.DataFrame(prep.transform(train[selected]), columns=selected)
    fig, axes = plt.subplots(2, len(show), figsize=(3.3 * len(show), 5.2))
    for j, c in enumerate(show):
        raw = train[c]
        axes[0, j].hist(raw, bins=35, color="#86b6ef", edgecolor=cfg.SURFACE, lw=.4)
        axes[0, j].set_title(f"{c} (cleaned)", fontsize=9)
        axes[0, j].text(.98, .92, f"mean {raw.mean():.2f}\nSD {raw.std():.2f}", transform=axes[0, j].transAxes,
                        ha="right", va="top", fontsize=7.5)
        axes[1, j].hist(Xt[c], bins=35, color="#1c5cab", edgecolor=cfg.SURFACE, lw=.4)
        axes[1, j].set_title(f"{c} (z-scored)", fontsize=9)
        axes[1, j].text(.98, .92, f"mean {Xt[c].mean():.2f}\nSD {Xt[c].std():.2f}", transform=axes[1, j].transAxes,
                        ha="right", va="top", fontsize=7.5)
        for a in axes[:, j]:
            a.set_yticks([])
    fig.suptitle("z-score normalisation (after IQR capping and imputation): mean 0, SD 1", x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "14_zscore_normalisation")


def plot_correlation_vif(train, sel):
    X = train[cfg.CLEAN_FEATURES]
    c = X.corr(method="spearman")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
    im = a1.imshow(c.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
    a1.set_xticks(range(len(c)), c.columns); a1.set_yticks(range(len(c)), c.columns); a1.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            if i != j and abs(c.values[i, j]) >= .3:
                a1.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(c.values[i, j]) > .65 else cfg.TEXT_PRIMARY)
    for t in a1.get_xticklabels() + a1.get_yticklabels():
        if t.get_text() not in sel["selected"]:
            t.set_color(cfg.NEUTRAL)
    fig.colorbar(im, ax=a1, fraction=.04, label="Spearman rho")
    a1.set_title("Spearman correlation (grey labels = not selected)")
    tab = sel["table"].set_index("feature")
    v_all = vif(X).replace(np.inf, 1e6).sort_values()
    a2.barh(v_all.index, np.log10(v_all.values), color=np.where(v_all.values > cfg.VIF_MAX, "#e34948", "#86b6ef"),
            height=.6)
    a2.axvline(np.log10(cfg.VIF_MAX), color=cfg.TEXT_SECONDARY, ls="--", lw=1)
    a2.axvline(1, color=cfg.NEUTRAL, ls=":", lw=1)
    a2.set_xlabel("log10 VIF (all 12 cleaned features; 6 = infinite)"); a2.grid(axis="y", visible=False)
    a2.set_title("Variance inflation: red = exact linear redundancy")
    del tab
    fig.tight_layout()
    return save_fig(fig, "15_correlation_and_vif")


def plot_feature_selection(t, comp=None, best=None):
    s = t.sort_values("mutual_info")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax.barh(s.feature, s.mutual_info, color=np.where(s.selected, "#1c5cab", cfg.NEUTRAL), height=.6)
    ax.axvline(cfg.MIN_MUTUAL_INFO, color="#eb6834", ls="--", lw=1)
    for i, (v, r) in enumerate(zip(s.mutual_info, s.reason)):
        ax.text(v + .001, i, r.replace("dropped: ", ""), va="center", fontsize=7.5, color=cfg.TEXT_SECONDARY)
    ax.set_xlabel("Mutual information with genre (training split)"); ax.grid(axis="y", visible=False)
    ax.set_title(f"Rules per feature ({int(s.selected.sum())} of {len(s)} kept: dark blue)", fontsize=10)
    strat = comp.strategy.unique()
    yy = np.arange(len(strat)); h = .36
    h = .27
    for i, (ref, col) in enumerate(zip(comp.reference_model.unique(), ["#e87ba4", "#4a3aa7", "#008300"])):
        d = comp[comp.reference_model == ref].set_index("strategy").loc[strat]
        ax2.barh(yy + (i - 1) * h, d.score, h - .03, xerr=d.sd, color=col, label=ref,
                 error_kw=dict(ecolor=cfg.NEUTRAL, lw=.8))
    ax2.set_yticks(yy, [f"{x} ({comp[comp.strategy == x].n_features.iloc[0]})" for x in strat])
    for lab in ax2.get_yticklabels():
        if lab.get_text().startswith(best):
            lab.set_fontweight("bold")
    ax2.set_xlim(.5, .95); ax2.set_xlabel("Repeated 5-fold CV score = worst of the 5 weighted metrics")
    ax2.axvline(cfg.TARGET_THRESHOLD, color="#e34948", ls="--", lw=1)
    ax2.legend(loc="lower right", fontsize=7.5); ax2.grid(axis="y", visible=False)
    ax2.set_title(f"Strategy comparison: best = {best}", fontsize=10)
    fig.suptitle("Feature selection: rule-based filters, then the strategy is chosen by cross-validation",
                 x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "16_feature_selection")


def run() -> dict:
    cfg.ensure_dirs()
    train, lab = load("train"), load("labelled")
    spec = load_json(cfg.CLEANING_SPEC_FILE)
    sel = select_features(train)
    sel["table"].to_csv(cfg.TABLES_DIR / "feature_selection.csv", index=False)
    sel["comparison"].to_csv(cfg.TABLES_DIR / "feature_strategy_comparison.csv", index=False)
    save_json({"selected_features": sel["selected"], "strategy": sel["strategy"],
               "strategies": sel["strategies"], "dropped_vif": sel["dropped_vif"],
               "latent_basis": sel["latent_basis"]}, cfg.SELECTED_FEATURES_FILE)
    imp = plot_missing_strategy(lab, spec); imp.to_csv(cfg.TABLES_DIR / "imputation_comparison_J.csv", index=False)
    ot = outlier_table(train); ot.to_csv(cfg.TABLES_DIR / "outlier_summary.csv", index=False)
    plot_outliers(train, ot); plot_zscore(train, sel["selected"]); plot_correlation_vif(train, sel)
    plot_feature_selection(sel["table"], sel["comparison"], sel["strategy"])
    log.info("Strategy comparison:\n%s", sel["comparison"].groupby("strategy")[["n_features", "score"]].max().round(4))
    log.info("Chosen %s -> %d features: %s", sel["strategy"], len(sel["selected"]), sel["selected"])
    return {"selection": sel, "imputation": imp, "outliers": ot}


if __name__ == "__main__":
    run()

"""
Stage 2: exploratory data analysis and data-quality evidence.

Questions answered (figures 01-11, tables eda_*.csv)
-----------------------------------------------------
* How balanced are the genres?                                      (01)
* What is missing, and which columns are duplicates?                (02, 03)
* Is the extreme range of A a unit error?                           (04)
* How does each feature differ by genre?                            (05, 09)
* How are features correlated, and which are exact linear
  combinations of others?                                           (06, 07)
* Do the unlabelled shows come from the same distribution as the
  labelled ones (dataset drift)?                                    (08)
* How separable are the genres in 2-D?                              (10, 11)

Run:  python -m src.analysis.eda
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src.analysis.stats import bh_fdr
from src.data.make_dataset import load, load_raw
from src.utils import get_logger
from src.visualization.visualize import DIV_CMAP, SEQ_CMAP, save_fig

log = get_logger(__name__)
G = cfg.CLASSES


def plot_class_distribution(lab):
    c = lab[cfg.TARGET].value_counts().reindex(G)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(G, c.values, color=[cfg.COLORS[g] for g in G], width=.6)
    for i, v in enumerate(c.values):
        ax.text(i, v + 6, f"{v} ({v / c.sum():.0%})", ha="center", fontsize=9)
    ax.set_ylim(0, c.max() * 1.15); ax.set_ylabel("Shows"); ax.grid(axis="x", visible=False)
    ax.set_title("Genre distribution (600 labelled shows): imbalanced, Comedy dominates")
    return save_fig(fig, "01_genre_distribution", "Class imbalance -> macro-averaged metrics and class weights are used")


def plot_quality(tr, te):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    miss = pd.DataFrame({"Training file": tr[cfg.RAW_FEATURES].isna().mean() * 100,
                         "Test file": te[cfg.RAW_FEATURES].isna().mean() * 100})
    x = np.arange(len(miss)); w = .38
    a1.bar(x - w / 2, miss["Training file"], w - .03, color="#2a78d6", label="Training file")
    a1.bar(x + w / 2, miss["Test file"], w - .03, color="#eb6834", label="Test file")
    a1.set_xticks(x, miss.index); a1.set_ylabel("% missing"); a1.legend(); a1.grid(axis="x", visible=False)
    a1.set_title("Missing values by raw column")
    both = pd.concat([tr, te])[cfg.RAW_FEATURES]
    eq = pd.DataFrame(index=cfg.RAW_FEATURES, columns=cfg.RAW_FEATURES, dtype=float)
    for a in cfg.RAW_FEATURES:
        for b in cfg.RAW_FEATURES:
            m = both[a].notna() & both[b].notna()
            eq.loc[a, b] = np.isclose(both.loc[m, a], both.loc[m, b]).mean()
    im = a2.imshow(eq.values, cmap=SEQ_CMAP, vmin=0, vmax=1)
    a2.set_xticks(range(15), cfg.RAW_FEATURES); a2.set_yticks(range(15), cfg.RAW_FEATURES); a2.grid(False)
    for i in range(15):
        for j in range(15):
            if i != j and eq.values[i, j] > .99:
                a2.text(j, i, "=", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=a2, fraction=.04, label="Share of identical values")
    a2.set_title("Identical-value matrix: D, F, I, K are one feature")
    fig.tight_layout()
    return save_fig(fig, "02_data_quality_missing_and_duplicates")


def plot_duplicates(tr):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col in zip(axes, ["D", "F", "K"]):
        m = tr[col].notna()
        ax.scatter(tr.loc[m, "I"], tr.loc[m, col], s=10, color="#2a78d6", alpha=.6, edgecolor="none")
        lim = [tr.I.min(), tr.I.max()]
        ax.plot(lim, lim, color=cfg.NEUTRAL, ls="--", lw=1)
        ax.set_xlabel("I"); ax.set_ylabel(col)
        ax.set_title(f"{col} vs I: r = {tr.loc[m, [col, 'I']].corr().iloc[0, 1]:.3f}, "
                     f"{tr[col].isna().sum()} missing", fontsize=10)
    fig.suptitle("Duplicate columns: D, F and K repeat I exactly (I is complete, so it is kept)", x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "03_duplicate_columns")


def plot_a_unit_error(tr, te, lab):
    both = pd.concat([tr.assign(file="Training"), te.assign(file="Test")])
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    ax = axes[0]
    a = both["A"].abs().clip(lower=1e-3)
    ax.hist(np.log10(a), bins=60, color="#86b6ef", edgecolor=cfg.SURFACE)
    ax.axvspan(np.log10(5.3), np.log10(12.1), color="#eb6834", alpha=.15)
    ax.text(np.log10(8), ax.get_ylim()[1] * .8, "empty gap\n5.2 - 12.2", ha="center", fontsize=8, color="#b52f2f")
    ax.set_xlabel("log10 |A| (both files)"); ax.set_ylabel("Shows"); ax.set_title("Raw |A|: two clusters 100x apart")
    ax = axes[1]
    big = both["A"].abs() >= cfg.A_SCALE_THRESHOLD
    bins = np.linspace(-5, 6, 40)
    ax.hist(both.loc[~big, "A"], bins=bins, density=True, histtype="step", lw=2, color="#2a78d6",
            label=f"Normal values (n={(~big).sum()})")
    ax.hist(both.loc[big, "A"] / 100, bins=bins, density=True, histtype="step", lw=2, color="#eb6834",
            label=f"Large values / 100 (n={big.sum()})")
    ks = stats.ks_2samp(both.loc[~big, "A"], both.loc[big, "A"] / 100)
    ax.legend(fontsize=7.5); ax.set_yticks([]); ax.set_xlabel("A")
    ax.set_title(f"Rescaled values match (KS p = {ks.pvalue:.2f})")
    ax = axes[2]
    for g in G:
        ax.hist(lab.loc[lab[cfg.TARGET] == g, "A"], bins=30, density=True, histtype="step", lw=2,
                color=cfg.COLORS[g], label=g)
    ax.legend(); ax.set_yticks([]); ax.set_xlabel("A after correction")
    ax.set_title("Corrected A by genre (Reality lower)")
    from src.data.make_dataset import a_ratio_check
    r = a_ratio_check(tr, te)
    ax = axes[3]
    ax.hist(r["ratios"], bins=20, color="#1c5cab", edgecolor=cfg.SURFACE)
    ax.axvline(100, color="#e34948", ls="--", lw=1.2)
    ax.set_xlabel("raw A / A predicted from C, I, L, M, N"); ax.set_ylabel("Flagged shows")
    ax.set_title(f"Independent check: median ratio {r['median']:.1f}")
    fig.suptitle("Feature A: ~5% of values carry a x100 unit error; corrected by dividing by 100 and flagged",
                 x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "04_feature_A_unit_error")


def plot_distributions(lab):
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), cfg.CLEAN_FEATURES):
        data = [lab.loc[lab[cfg.TARGET] == g, c] for g in G]
        parts = ax.violinplot(data, showextrema=False, widths=.8)
        for pc, g in zip(parts["bodies"], G):
            pc.set_facecolor(cfg.COLORS[g]); pc.set_alpha(.3); pc.set_edgecolor("none")
        bp = ax.boxplot(data, widths=.16, patch_artist=True, showfliers=False, medianprops=dict(color="white", lw=1.4))
        for p, g in zip(bp["boxes"], G):
            p.set_facecolor(cfg.COLORS[g]); p.set_edgecolor(cfg.COLORS[g])
        p = stats.kruskal(*data).pvalue
        ax.set_xticks([1, 2, 3], G, fontsize=8); ax.grid(axis="x", visible=False)
        ax.set_title(f"{c}  (Kruskal p = {p:.1e})", fontsize=9.5)
    fig.suptitle("Cleaned features by genre (labelled shows)", x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "05_feature_distributions_by_genre")


def plot_correlation(tr, lab):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, d, cols, title in [(a1, tr, cfg.RAW_FEATURES, "Raw columns A-O (Pearson)"),
                               (a2, lab, cfg.CLEAN_FEATURES, "Cleaned features (Pearson)")]:
        c = d[cols].corr()
        im = ax.imshow(c.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(cols)), cols); ax.grid(False)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = c.values[i, j]
                if i != j and abs(v) >= .3:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(v) > .65 else cfg.TEXT_PRIMARY)
        ax.set_title(title)
    fig.colorbar(im, ax=[a1, a2], fraction=.02, label="r")
    fig.suptitle("Correlation analysis: a dense block (B, C, I, J, L, M, N); A, E, G, H, O almost uncorrelated",
                 x=.01, ha="left")
    return save_fig(fig, "06_correlation_heatmaps")


def plot_linear_dependency(lab, spec):
    block = ["B", "C", "I", "J", "L", "M", "N"]
    Z = lab[block] - lab[block].mean()
    sv = np.linalg.svd(Z, compute_uv=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    axes[0].bar(range(1, 8), sv / sv.max(), color=["#2a78d6"] * 5 + ["#e34948"] * 2, width=.6)
    axes[0].set_xlabel("Singular value #"); axes[0].set_ylabel("Relative size")
    axes[0].set_title("B, C, I, J, L, M, N span only 5 dimensions")
    for i, v in enumerate(sv / sv.max()):
        axes[0].text(i + 1, v + .02, f"{v:.3f}", ha="center", fontsize=7.5)
    obs = lab[lab.J_was_reconstructed == 0]
    pred = spec["j_intercept"] + sum(obs[c] * w for c, w in spec["j_coef"].items())
    axes[1].scatter(obs.J, pred, s=8, color="#2a78d6", alpha=.6)
    axes[1].plot([obs.J.min(), obs.J.max()], [obs.J.min(), obs.J.max()], color=cfg.NEUTRAL, ls="--", lw=1)
    axes[1].set_xlabel("Observed J"); axes[1].set_ylabel("J from B, C, I, L, M, N")
    axes[1].set_title(f"J is exact (max residual {spec['j_max_residual']:.0e})")
    coef = pd.Series(spec["j_coef"])
    axes[2].barh(coef.index, coef.values, color="#1c5cab", height=.55)
    axes[2].axvline(0, color=cfg.NEUTRAL, lw=1); axes[2].grid(axis="y", visible=False)
    axes[2].set_title("J = linear combination (coefficients)"); axes[2].set_xlabel("Weight")
    fig.suptitle("Exact linear redundancy: 2 of 7 block features carry no new information; missing J is recoverable",
                 x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "07_linear_dependency_and_J_reconstruction")


def drift_table(lab, unl):
    rows = []
    for c in cfg.CLEAN_FEATURES:
        k = stats.ks_2samp(lab[c], unl[c])
        rows.append({"feature": c, "ks_stat": k.statistic, "p": k.pvalue,
                     "labelled_mean": lab[c].mean(), "unlabelled_mean": unl[c].mean()})
    t = pd.DataFrame(rows)
    t["q"] = bh_fdr(t["p"].values)
    return t


def plot_drift(t, lab, unl):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    s = t.sort_values("ks_stat")
    a1.barh(s.feature, s.ks_stat, color=np.where(s.q < .05, "#e34948", "#86b6ef"), height=.6)
    for i, (v, q) in enumerate(zip(s.ks_stat, s.q)):
        a1.text(v + .002, i, f"q={q:.2f}", va="center", fontsize=7.5, color=cfg.TEXT_SECONDARY)
    a1.set_xlabel("KS statistic (labelled vs unlabelled shows)"); a1.grid(axis="y", visible=False)
    a1.set_title("Distribution shift per feature (red = FDR q < 0.05)")
    worst = t.sort_values("ks_stat").feature.iloc[-1]
    a2.hist(lab[worst], bins=35, density=True, histtype="step", lw=2, color="#2a78d6", label="Labelled (600)")
    a2.hist(unl[worst], bins=35, density=True, histtype="step", lw=2, color="#eb6834", label="Unlabelled (400)")
    a2.legend(); a2.set_yticks([]); a2.set_xlabel(worst)
    a2.set_title(f"Largest shift: {worst}")
    fig.tight_layout()
    return save_fig(fig, "08_train_vs_test_drift", "After FDR correction no feature shifts significantly; adversarial validation in the stress tests confirms")


def separation_table(lab):
    y = lab[cfg.TARGET]
    rows = []
    mi = mutual_info_classif(lab[cfg.CLEAN_FEATURES], y, random_state=cfg.RANDOM_STATE)
    for c, m in zip(cfg.CLEAN_FEATURES, mi):
        k = stats.kruskal(*[lab.loc[y == g, c] for g in G])
        rows.append({"feature": c, "kruskal_H": k.statistic, "p": k.pvalue, "mutual_info": m,
                     **{f"mean_{g}": lab.loc[y == g, c].mean() for g in G}})
    t = pd.DataFrame(rows)
    t["q"] = bh_fdr(t["p"].values)
    return t.sort_values("mutual_info", ascending=False)


def plot_separation(t):
    s = t.sort_values("mutual_info")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    col = np.where(s.q < .05, "#1c5cab", cfg.NEUTRAL)
    a1.barh(s.feature, s.mutual_info, color=col, height=.6); a1.set_xlabel("Mutual information with genre")
    a1.grid(axis="y", visible=False); a1.set_title("Information about genre")
    a2.barh(s.feature, -np.log10(s.q.clip(lower=1e-300)), color=col, height=.6)
    a2.axvline(-np.log10(.05), color="#eb6834", ls="--", lw=1)
    a2.set_xlabel("-log10 FDR q (Kruskal-Wallis)"); a2.grid(axis="y", visible=False); a2.set_yticks([])
    a2.set_title("Statistical difference between genres")
    fig.suptitle("Univariate separation (dark blue = significant): E, H, O carry no genre signal", x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "09_univariate_separation")


def plot_projection(lab):
    X = StandardScaler().fit_transform(lab[cfg.CLEAN_FEATURES]); y = lab[cfg.TARGET]
    pca = PCA(2, random_state=cfg.RANDOM_STATE).fit(X); P = pca.transform(X)
    L = LinearDiscriminantAnalysis(n_components=2).fit(X, y).transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, Z, t in [(axes[0], P, f"PCA (explains {pca.explained_variance_ratio_.sum():.0%} of variance)"),
                     (axes[1], L, "LDA (supervised projection)")]:
        for g in G:
            m = (y == g).values
            ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=.6, color=cfg.COLORS[g], label=g, edgecolor="none")
        ax.legend(markerscale=1.6); ax.set_title(t, fontsize=10); ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    fig.suptitle("2-D projections: genres overlap linearly, so non-linear models are needed", x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "10_pca_lda_projection")


def plot_pairs(lab, top):
    k = len(top)
    fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))
    for i, a in enumerate(top):
        for j, b in enumerate(top):
            ax = axes[i, j]
            for g in G:
                s = lab[lab[cfg.TARGET] == g]
                if i == j:
                    ax.hist(s[a], bins=25, density=True, histtype="step", lw=1.6, color=cfg.COLORS[g])
                else:
                    ax.scatter(s[b], s[a], s=5, alpha=.5, color=cfg.COLORS[g], label=g, edgecolor="none")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(a)
            if i == k - 1:
                ax.set_xlabel(b)
    axes[0, 1].legend(markerscale=2, fontsize=7, loc="upper left")
    fig.suptitle(f"Pair plot of the {k} most informative features", x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "11_pairplot_top_features")


def run() -> dict:
    cfg.ensure_dirs()
    tr, te = load_raw()
    lab, unl = load("labelled"), pd.read_csv(cfg.UNLABELLED_FILE)
    from src.utils import load_json
    spec = load_json(cfg.CLEANING_SPEC_FILE)
    desc = lab.groupby(cfg.TARGET)[cfg.CLEAN_FEATURES].agg(["mean", "std"]).T.round(3)
    desc.to_csv(cfg.TABLES_DIR / "eda_descriptive_by_genre.csv")
    sep = separation_table(lab); sep.to_csv(cfg.TABLES_DIR / "eda_univariate_separation.csv", index=False)
    drift = drift_table(lab, unl); drift.to_csv(cfg.TABLES_DIR / "eda_train_test_drift.csv", index=False)
    plot_class_distribution(lab); plot_quality(tr, te); plot_duplicates(tr); plot_a_unit_error(tr, te, lab)
    plot_distributions(lab); plot_correlation(tr, lab); plot_linear_dependency(lab, spec)
    plot_drift(drift, lab, unl); plot_separation(sep); plot_projection(lab)
    plot_pairs(lab, sep.feature.head(4).tolist())
    log.info("EDA done. Top features by MI: %s | drifted (q<0.05): %s", sep.feature.head(5).tolist(),
             drift.loc[drift.q < .05, "feature"].tolist())
    return {"separation": sep, "drift": drift, "descriptive": desc}


if __name__ == "__main__":
    run()

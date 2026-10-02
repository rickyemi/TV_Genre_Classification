"""
Stage 5: evaluate the classifiers on Train and Test.

Two evaluation designs are reported for every model:
    Holdout   Train = the 480-show training split (in-sample), Test = the 120
              held-out shows (never used for tuning).
    CV        repeated stratified 5-fold CV on all 600 labelled shows
              (5 repeats = 25 refits, hyperparameters fixed): Train = in-fold
              fit, Test = out-of-fold predictions. Every show is a Test show
              once per repeat, so this estimate rests on 600 shows instead of
              120 (Test accuracy standard error about 0.012 instead of 0.029).

Metrics (see src/models/metrics.py): Accuracy, and one-vs-rest Sensitivity,
Specificity, Precision, F1 (plus NPV, AUC, log-loss, Brier) per genre,
averaged by genre share (weighted = the >= 0.89 target) and unweighted
(macro, secondary).

Outputs: comparison tables (CSV + Markdown), per-genre table, acceptance
check, confusion matrices, ROC curves, metric comparison, calibration,
XGBoost gain and permutation importance (figures 19-25).

Run:  python -m src.models.evaluate_model
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from src import config as cfg
from src.data.make_dataset import get_xy, load
from src.models.metrics import summary_metrics
from src.models.train_model import load_models
from src.utils import get_logger, load_json
from src.visualization.visualize import SEQ_CMAP, save_fig

log = get_logger(__name__)
K = len(cfg.CLASSES)
TARGET = cfg.TARGET_METRICS
ALL_MODELS = cfg.MODEL_NAMES + cfg.SCREENED_MODELS


def multiclass_metrics(y, P) -> dict:
    """Weighted + macro one-vs-rest metrics, AUC and probability scores from a probability matrix."""
    y = np.asarray(y); P = np.asarray(P)
    out = summary_metrics(y, P.argmax(1))
    auc = [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(K)]
    w = np.bincount(y, minlength=K) / len(y)
    out["AUC"], out["AUC_macro"] = float(w @ auc), float(np.mean(auc))
    for k, c in enumerate(cfg.CLASSES):
        out["per_class"][c]["AUC"] = auc[k]
    out["LogLoss"] = log_loss(y, P, labels=range(K))
    out["Brier"] = float(np.mean(np.sum((P - np.eye(K)[y]) ** 2, axis=1)))
    return out


def evaluate(models, sets):
    res, probs = {}, {}
    for name, m in models.items():
        for s, (X, y) in sets.items():
            P = m.predict_proba(X)
            probs[(name, s)] = P
            res[(name, s)] = multiclass_metrics(y, P)
    return res, probs


def repeated_cv(models, X, y, repeats=cfg.CV_REPEATS):
    """In-fold Train metrics (mean over folds) and pooled out-of-fold Test metrics (mean over repeats)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=cfg.RANDOM_STATE)
    splits = list(cv.split(X, y)); yv = np.asarray(y)
    res, folds = {}, []
    for name, m in models.items():
        tr_rows, te_rows, oof = [], [], np.zeros((len(yv), K))
        for i, (a, b) in enumerate(splits):
            mm = clone(m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mm.fit(X.iloc[a], y.iloc[a])
            r_tr = multiclass_metrics(yv[a], mm.predict_proba(X.iloc[a]))
            oof[b] = mm.predict_proba(X.iloc[b])
            r_fold = multiclass_metrics(yv[b], oof[b])
            folds.append({"Model": name, "fold": i, **{f"Train_{k}": r_tr[k] for k in TARGET},
                          **{f"Test_{k}": r_fold[k] for k in TARGET}})
            tr_rows.append(r_tr)
            if (i + 1) % 5 == 0:                      # end of one repeat: pooled OOF metrics
                te_rows.append(multiclass_metrics(yv, oof)); oof = np.zeros((len(yv), K))
        res[(name, "CV Train")] = _mean_metrics(tr_rows)
        res[(name, "CV Test")] = _mean_metrics(te_rows)
    return res, pd.DataFrame(folds)


def _mean_metrics(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out.update({f"{k}_sd": float(np.std([r[k] for r in rows])) for k in TARGET})
    out["per_class"] = {c: {m: float(np.mean([r["per_class"][c][m] for r in rows])) for m in rows[0]["per_class"][c]}
                        for c in cfg.CLASSES}
    out["confusion"] = np.mean([r["confusion"] for r in rows], axis=0)
    return out


METRIC_COLS = TARGET + ["NPV", "AUC"] + [f"{m}_macro" for m in ["Sensitivity", "Specificity", "Precision", "F1",
                                                                  "NPV"]] + ["AUC_macro", "LogLoss", "Brier"]


def comparison_table(res) -> pd.DataFrame:
    rows = []
    for (m, s), r in res.items():
        rows.append({"Model": m, "Set": s, **{k: r.get(k, np.nan) for k in METRIC_COLS}})
    return pd.DataFrame(rows)


def per_class_table(res, split="Test") -> pd.DataFrame:
    rows = []
    for (m, s), r in res.items():
        if s == split:
            for c, v in r["per_class"].items():
                rows.append({"Model": m, "Genre": c, **v})
    return pd.DataFrame(rows)


def to_markdown(tab, sets=("Train", "Test"), models=None, macro=True) -> str:
    models = models or cfg.MODEL_NAMES
    cols = [(m, s) for m in models for s in sets]
    lines = ["| Metric | " + " | ".join(f"{m} {s}" for m, s in cols) + " |", "|---|" + "---:|" * len(cols)]
    rows = [(k, k) for k in TARGET]
    if macro:
        rows += [(f"{k} (macro)", f"{k}_macro") for k in ["Sensitivity", "Specificity", "Precision", "F1"]]
    for label, met in rows:
        v = [tab[(tab.Model == m) & (tab.Set == s)][met].iloc[0] for m, s in cols]
        lines.append(f"| {label} | " + " | ".join(
            (f"**{x:.3f}**" if (met in TARGET and x >= cfg.TARGET_THRESHOLD) else f"{x:.3f}") for x in v) + " |")
    return "\n".join(lines)


def acceptance(res) -> pd.DataFrame:
    """Every weighted target metric >= 0.89 on Train and Test, for both evaluation designs; accuracy gap <= 0.05."""
    rows = []
    for m in cfg.MODEL_NAMES:
        for design, (tr, te) in {"Holdout": ("Train", "Test"), "CV": ("CV Train", "CV Test")}.items():
            for s in (tr, te):
                for k in TARGET:
                    v = res[(m, s)][k]
                    rows.append({"Model": m, "Design": design, "Set": s, "Criterion": k, "Value": v,
                                 "Rule": f">= {cfg.TARGET_THRESHOLD}", "Pass": bool(v >= cfg.TARGET_THRESHOLD)})
            gap = res[(m, tr)]["Accuracy"] - res[(m, te)]["Accuracy"]
            rows.append({"Model": m, "Design": design, "Set": "Train - Test", "Criterion": "Accuracy gap",
                         "Value": gap, "Rule": f"<= {cfg.ACCEPTANCE['max_train_test_gap']}",
                         "Pass": bool(gap <= cfg.ACCEPTANCE["max_train_test_gap"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def plot_confusion(res):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, m in zip(axes, cfg.MODEL_NAMES):
        cm = res[(m, "Test")]["confusion"]
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cm[i, j]}\n({rn[i, j]:.0%})", ha="center", va="center", fontsize=9.5,
                        color="white" if rn[i, j] > .55 else cfg.TEXT_PRIMARY)
        ax.set_xticks(range(K), cfg.CLASSES); ax.set_yticks(range(K), cfg.CLASSES); ax.grid(False)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{m} (Test, accuracy {res[(m, 'Test')]['Accuracy']:.1%})", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "19_confusion_matrices", "Held-out shows (n = 120); row percentages = per-genre sensitivity")


def plot_roc(probs, sets):
    y = np.asarray(sets["Test"][1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, (ax, c) in enumerate(zip(axes, cfg.CLASSES)):
        for m in cfg.MODEL_NAMES:
            p = probs[(m, "Test")][:, k]
            fpr, tpr, _ = roc_curve(y == k, p)
            ax.plot(fpr, tpr, color=cfg.MODEL_COLORS[m], label=f"{m} (AUC {roc_auc_score(y == k, p):.3f})")
        ax.plot([0, 1], [0, 1], color=cfg.NEUTRAL, ls="--", lw=1)
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity"); ax.legend(loc="lower right")
        ax.set_title(f"{c} vs rest (Test)", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "20_roc_curves_one_vs_rest")


def plot_metric_comparison(tab):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    x = np.arange(len(TARGET)); w = .26
    for ax, (tr, te, title) in zip(axes, [("Train", "Test", "Holdout: Train 480 / Test 120 shows"),
                                          ("CV Train", "CV Test", "Repeated 5x5 CV on all 600 shows")]):
        for i, m in enumerate(cfg.MODEL_NAMES):
            vte = tab[(tab.Model == m) & (tab.Set == te)][TARGET].iloc[0].values
            vtr = tab[(tab.Model == m) & (tab.Set == tr)][TARGET].iloc[0].values
            ax.bar(x + (i - 1) * w, vte, w - .03, color=cfg.MODEL_COLORS[m], label=f"{m} Test")
            ax.scatter(x + (i - 1) * w, vtr, marker="_", s=180, color=cfg.TEXT_PRIMARY, lw=2,
                       label="Train" if i == 0 else None, zorder=3)
            for xx, v in zip(x + (i - 1) * w, vte):
                ax.text(xx, .802, f"{v:.3f}", rotation=90, ha="center", va="bottom", fontsize=7, color="white")
        ax.axhline(cfg.TARGET_THRESHOLD, color="#e34948", ls="--", lw=1)
        ax.set_xticks(x, TARGET); ax.set_ylim(.8, 1.0); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Weighted one-vs-rest metric")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", fontsize=8, frameon=False)
    fig.suptitle("Target metrics: Test bars, Train markers, red dashed line = 0.89 target", x=.01, ha="left")
    fig.tight_layout(rect=(0, .06, 1, 1))
    return save_fig(fig, "21_metric_comparison")


def plot_per_class(pc):
    mets = ["Sensitivity", "Specificity", "Precision", "NPV", "F1", "AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, m in zip(axes, cfg.MODEL_NAMES):
        t = pc[pc.Model == m].set_index("Genre")[mets].loc[cfg.CLASSES]
        ax.imshow(t.values, cmap=SEQ_CMAP, vmin=.4, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(len(mets)):
                v = t.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > .8 else cfg.TEXT_PRIMARY)
        ax.set_xticks(range(len(mets)), mets, fontsize=8, rotation=20); ax.set_yticks(range(3), cfg.CLASSES)
        ax.grid(False); ax.set_title(m, fontsize=10)
    fig.suptitle("Per-genre metrics, holdout Test (one-vs-rest): Reality is the hardest genre to recall",
                 x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "22_per_genre_metrics")


def plot_calibration(probs, sets):
    y = np.asarray(sets["Test"][1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = np.linspace(1 / 3, 1, 6)
    for m in cfg.MODEL_NAMES:
        P = probs[(m, "Test")]; conf = P.max(1); right = (P.argmax(1) == y)
        idx = np.digitize(conf, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            s = idx == b
            if s.sum() >= 5:
                xs.append(conf[s].mean()); ys.append(right[s].mean())
        a1.plot(xs, ys, marker="o", ms=5, color=cfg.MODEL_COLORS[m], label=m)
        a2.hist(conf, bins=np.linspace(1 / 3, 1, 15), histtype="step", lw=2, color=cfg.MODEL_COLORS[m], label=m)
    a1.plot([1 / 3, 1], [1 / 3, 1], color=cfg.NEUTRAL, ls="--", lw=1)
    a1.set_xlabel("Predicted probability of the chosen genre"); a1.set_ylabel("Observed accuracy")
    a1.set_title("Top-label calibration (Test)", fontsize=10); a1.legend()
    a2.axvline(cfg.REVIEW_CONFIDENCE, color="#e34948", ls="--", lw=1)
    a2.text(cfg.REVIEW_CONFIDENCE, a2.get_ylim()[1] * .9, " manual-review\n threshold", fontsize=7.5, color="#b52f2f")
    a2.set_xlabel("Max predicted probability"); a2.set_ylabel("Shows"); a2.legend()
    a2.set_title("Confidence distribution (Test)", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "23_calibration_and_confidence")


def xgb_feature_names(model, feats):
    g = model.named_steps["gmm"]
    names = list(feats) if g.keep_input else []
    names += [f"logP({c})" for c in cfg.CLASSES]
    if g.components:
        k = g.gmm_.n_components
        names += [f"comp {cfg.CLASSES[c]}-{j + 1}" for c in range(K) for j in range(k)]
    return names


def plot_xgb_gain(model, feats):
    names = xgb_feature_names(model, feats)
    g = pd.Series(model.named_steps["clf"].get_booster().get_score(importance_type="gain"))
    g.index = [names[int(k[1:])] if k[1:].isdigit() else k for k in g.index]
    g = (g / g.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, .28 * len(g) + 1.4))
    ax.barh(g.index, g.values, color=[cfg.MODEL_COLORS["XGBoost"] if n in feats else "#86b6ef" for n in g.index],
            height=.6)
    ax.set_xlabel("Share of total gain"); ax.grid(axis="y", visible=False)
    ax.set_title("XGBoost gain: mixture log-posteriors (light) vs raw inputs (dark)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "24_xgboost_gain_importance")
    return g


def permutation_table(models, X, y):
    rows = []
    for m, mod in models.items():
        r = permutation_importance(mod, X, y, scoring="accuracy", n_repeats=20, random_state=cfg.RANDOM_STATE)
        for f, mu, sd in zip(X.columns, r.importances_mean, r.importances_std):
            rows.append({"Model": m, "feature": f, "accuracy_drop_mean": mu, "accuracy_drop_sd": sd})
    return pd.DataFrame(rows)


def plot_permutation(perm):
    order = perm.groupby("feature").accuracy_drop_mean.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    yy = np.arange(len(order)); h = .26
    for i, m in enumerate(cfg.MODEL_NAMES):
        d = perm[perm.Model == m].set_index("feature").loc[order]
        ax.barh(yy + (1 - i) * h, d.accuracy_drop_mean, xerr=d.accuracy_drop_sd, height=h - .02,
                color=cfg.MODEL_COLORS[m], label=m, error_kw=dict(ecolor=cfg.NEUTRAL, lw=.8))
    ax.set_yticks(yy, order); ax.axvline(0, color=cfg.NEUTRAL, lw=1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in Test accuracy when the input is shuffled (20 repeats)"); ax.legend(loc="lower right")
    ax.set_title("Permutation importance of the 5 inputs, all three models")
    fig.tight_layout()
    return save_fig(fig, "25_permutation_importance")


def run() -> dict:
    cfg.ensure_dirs()
    meta = load_json(cfg.MODEL_METADATA_FILE); feats = meta["features"]
    models = load_models("evaluation", include_screened=True)
    sets = {"Train": get_xy(load("train"), feats), "Test": get_xy(load("holdout"), feats)}
    res, probs = evaluate(models, sets)
    log.info("repeated %dx5 CV on all labelled shows", cfg.CV_REPEATS)
    cv_res, folds = repeated_cv(models, *get_xy(load("labelled"), feats))
    res.update(cv_res)
    folds.to_csv(cfg.TABLES_DIR / "cv_fold_metrics.csv", index=False)
    tab = comparison_table(res)
    order = {s: i for i, s in enumerate(["Train", "Test", "CV Train", "CV Test"])}
    tab = tab.sort_values(["Model", "Set"], key=lambda c: c.map(order) if c.name == "Set" else c.map(
        {m: i for i, m in enumerate(ALL_MODELS)}))
    tab.round(5).to_csv(cfg.TABLES_DIR / "model_comparison_train_test.csv", index=False)
    md_hold = to_markdown(tab, ("Train", "Test")); md_cv = to_markdown(tab, ("CV Train", "CV Test"))
    md_scr = to_markdown(tab, ("Train", "Test", "CV Train", "CV Test"), models=cfg.SCREENED_MODELS, macro=False)
    (cfg.TABLES_DIR / "model_comparison_train_test.md").write_text(
        "## Holdout (Train 480 / Test 120)\n\n" + md_hold + "\n\n## Repeated 5x5 CV (600 shows)\n\n" + md_cv +
        "\n\n## Screened model\n\n" + md_scr + "\n")
    pc = per_class_table(res); pc.round(4).to_csv(cfg.TABLES_DIR / "per_genre_metrics_test.csv", index=False)
    per_class_table(res, "CV Test").round(4).to_csv(cfg.TABLES_DIR / "per_genre_metrics_cv_test.csv", index=False)
    acc = acceptance(res); acc.to_csv(cfg.TABLES_DIR / "acceptance_criteria.csv", index=False)
    cv = pd.DataFrame([{"Model": m, "deployed": v["deployed"], "tuning CV score (min weighted)": v["cv_score_min_weighted"],
                        "SD": v["cv_score_sd"], "tuning CV train score": v["cv_train_score"],
                        "gap guard met": v["gap_guard_met"]} for m, v in meta["models"].items()])
    cv.to_csv(cfg.TABLES_DIR / "cross_validation_summary.csv", index=False)
    main = {k: v for k, v in models.items() if k in cfg.MODEL_NAMES}
    plot_confusion(res); plot_roc(probs, sets); plot_metric_comparison(tab); plot_per_class(pc)
    plot_calibration(probs, sets)
    g = plot_xgb_gain(main["XGBoost"], feats); g.sort_values(ascending=False).to_csv(
        cfg.TABLES_DIR / "xgboost_gain_importance.csv", header=["gain_share"])
    perm = permutation_table(main, *sets["Test"]); perm.to_csv(cfg.TABLES_DIR / "permutation_importance.csv", index=False)
    plot_permutation(perm)
    log.info("Holdout:\n%s\nCV:\n%s\nScreened:\n%s", md_hold, md_cv, md_scr)
    for d in ["Holdout", "CV"]:
        a = acc[acc.Design == d]
        log.info("Acceptance %s: %d/%d checks pass | per model %s", d, a.Pass.sum(), len(a),
                 a.groupby("Model").Pass.all().to_dict())
    return {"results": res, "table": tab, "per_class": pc, "acceptance": acc, "cv": cv, "markdown": md_hold,
            "markdown_cv": md_cv, "markdown_screened": md_scr, "permutation": perm, "probs": probs, "sets": sets,
            "folds": folds}


if __name__ == "__main__":
    run()

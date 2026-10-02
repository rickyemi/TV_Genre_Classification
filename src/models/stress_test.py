"""
Stage 6: stress tests for the three genre classifiers.

 #  Test                       Question                                            Pass rule (fixed before running)
--  -------------------------  --------------------------------------------------  --------------------------------------
 1  Bootstrap CI (holdout)     How uncertain are Test metrics with 120 shows?      accuracy CI lower bound > 0.61
                                                                                   (always-Comedy baseline)
 2  Repeated CV (all 600)      Stable estimate without a single small holdout      CV accuracy SD <= 0.05
 3  Measurement noise          Gaussian noise of 0.25 training SD                  accuracy drop <= 0.05
 4  Missing data               +20% blank cells at inference                       accuracy drop <= 0.05
 5  Unit-error injection       5% of A values x100, raw vs with cleaning step      drop with cleaning <= 0.01
                               (A is not a model input since the 5-D basis was chosen, so raw errors in A
                               can no longer reach the classifiers; kept as a regression check)
 6  Adversarial validation     can a model tell labelled from unlabelled shows?    AUC <= 0.60 (no drift)
 7  Label noise (retrain)      10% of training genres flipped                      accuracy drop <= 0.05
 8  Permutation test           refit on shuffled genres                            real - max null macro-F1 >= 0.20
 9  Learning curve             25 / 50 / 75 / 100% of training shows               reported (does more data help?)
10  Label-quality audit        out-of-fold confident disagreements                 reported (shows to re-review)
11  Input edge cases           all-missing, extreme values, single row, repeats    finite probabilities summing to 1

Run:  python -m src.models.stress_test     (about 3 minutes)
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src import config as cfg
from src.data.make_dataset import get_xy, load
from src.models.metrics import summary_metrics
from src.models.train_model import fit_params, load_models
from src.utils import get_logger, load_json, set_seed
from src.visualization.visualize import save_fig

log = get_logger(__name__)
RNG = np.random.default_rng(cfg.RANDOM_STATE)
M = cfg.MODEL_NAMES


def acc(m, X, y):
    return accuracy_score(y, m.predict_proba(X).argmax(1))


def refit(m, name, X, y):
    mm = clone(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm.fit(X, y, **fit_params(name, np.asarray(y)))
    return mm


def bootstrap(models, X, y, n=1000):
    y = np.asarray(y); rows = []
    for name, m in models.items():
        P = m.predict_proba(X); pred = P.argmax(1); d = []
        for _ in range(n):
            i = RNG.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 3:
                continue
            sm = summary_metrics(y[i], pred[i])
            d.append({"Accuracy": sm["Accuracy"], "Specificity_w": sm["Specificity"], "F1_w": sm["F1"],
                      "F1_macro": f1_score(y[i], pred[i], average="macro"),
                      "AUC_macro": roc_auc_score(y[i], P[i], multi_class="ovr"),
                      **{f"Sens_{c}": np.mean(pred[i][y[i] == k] == k) for k, c in enumerate(cfg.CLASSES)}})
        d = pd.DataFrame(d)
        for c in d.columns:
            rows.append({"Model": name, "Metric": c, "mean": d[c].mean(), "ci_low": d[c].quantile(.025),
                         "ci_high": d[c].quantile(.975)})
    return pd.DataFrame(rows)


def repeated_cv(models, Xall, yall):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=cfg.RANDOM_STATE)
    rows = []
    for name, m in models.items():
        for k, (a, b) in enumerate(cv.split(Xall, yall)):
            mm = refit(m, name, Xall.iloc[a], yall.iloc[a])
            P = mm.predict_proba(Xall.iloc[b]); pred = P.argmax(1)
            rows.append({"Model": name, "fold": k, "Accuracy": accuracy_score(yall.iloc[b], pred),
                         "F1_macro": f1_score(yall.iloc[b], pred, average="macro"),
                         "AUC_macro": roc_auc_score(yall.iloc[b], P, multi_class="ovr")})
    return pd.DataFrame(rows)


def corruption(models, Xtr, X, y, feats):
    raw_hold = load("holdout")
    sd = Xtr.std()
    rows = []
    for name, m in models.items():
        base = acc(m, X, y)
        for lv in [0, .1, .25, .5]:
            Xn = X + RNG.normal(0, 1, X.shape) * sd.values * lv
            rows.append({"Model": name, "test": "noise_sd", "level": lv, "Accuracy": acc(m, Xn, y)})
        for lv in [0, .1, .2, .3]:
            rows.append({"Model": name, "test": "missing", "level": lv, "Accuracy": acc(m, X.mask(RNG.random(X.shape) < lv), y)})
        # unit error: corrupt 5% of A x100, then score raw (no cleaning) vs re-cleaned
        spec = load_json(cfg.CLEANING_SPEC_FILE)
        bad = RNG.random(len(raw_hold)) < .05
        corrupt = raw_hold.copy(); corrupt.loc[bad, "A"] = corrupt.loc[bad, "A"] * 100
        Xraw = corrupt[feats]
        recl = corrupt.copy()          # apply the same A rule the cleaning stage uses
        recl.loc[recl.A.abs() >= spec["a_threshold"], "A"] /= spec["a_factor"]
        rows.append({"Model": name, "test": "unit_error_no_cleaning", "level": .05, "Accuracy": acc(m, Xraw, y)})
        rows.append({"Model": name, "test": "unit_error_with_cleaning", "level": .05, "Accuracy": acc(m, recl[feats], y)})
        rows.append({"Model": name, "test": "baseline", "level": 0, "Accuracy": base})
    return pd.DataFrame(rows)


def adversarial(lab, unl, feats):
    X = pd.concat([lab[feats], unl[feats]], ignore_index=True)
    y = np.r_[np.zeros(len(lab)), np.ones(len(unl))]
    p = cross_val_predict(XGBClassifier(max_depth=3, n_estimators=200, learning_rate=.05, verbosity=0, n_jobs=1),
                          X, y, cv=StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def retrain_tests(models, Xtr, ytr, X, y):
    ln, perm, lc = [], [], []
    for name, m in models.items():
        base = acc(m, X, y)
        yn = ytr.copy(); flip = RNG.random(len(yn)) < .10
        yn[flip] = (yn[flip] + RNG.integers(1, 3, flip.sum())) % 3
        ln.append({"Model": name, "clean": base, "noisy": acc(refit(m, name, Xtr, yn), X, y)})
        for k in range(3):
            ys = pd.Series(RNG.permutation(ytr.values), index=ytr.index)
            mm = refit(m, name, Xtr, ys)
            perm.append({"Model": name, "perm": k,
                         "F1_real": f1_score(y, m.predict_proba(X).argmax(1), average="macro"),
                         "F1_null": f1_score(RNG.permutation(np.asarray(y)), mm.predict_proba(X).argmax(1), average="macro")})
        for frac in [.25, .5, .75, 1.0]:
            if frac < 1:
                idx = (pd.DataFrame({"y": ytr}).groupby("y", group_keys=False)
                       .apply(lambda g: g.sample(frac=frac, random_state=cfg.RANDOM_STATE)).index)
                mm = refit(m, name, Xtr.loc[idx], ytr.loc[idx])
            else:
                idx, mm = ytr.index, m
            lc.append({"Model": name, "fraction": frac, "n": len(idx), "Train_acc": acc(mm, Xtr.loc[idx], ytr.loc[idx]),
                       "Test_acc": acc(mm, X, y), "Test_F1": f1_score(y, mm.predict_proba(X).argmax(1), average="macro")})
    return pd.DataFrame(ln), pd.DataFrame(perm), pd.DataFrame(lc)


def label_audit(models, lab, feats):
    """Out-of-fold probabilities from all three models; flag confident disagreements."""
    X, y = get_xy(lab, feats)
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    P = np.zeros((len(lab), 3))
    for name, m in models.items():
        for a, b in cv.split(X, y):
            P[b] += refit(m, name, X.iloc[a], y.iloc[a]).predict_proba(X.iloc[b]) / len(models)
    out = lab[["show_id", cfg.TARGET]].copy()
    for k, c in enumerate(cfg.CLASSES):
        out[f"oof_P_{c}"] = P[:, k].round(3)
    out["oof_pred"] = [cfg.CLASSES[i] for i in P.argmax(1)]
    out["P_given_label"] = P[np.arange(len(P)), y.values].round(3)
    out["suspect_label"] = (out.oof_pred != out[cfg.TARGET]) & (P.max(1) >= .80)
    return out.sort_values("P_given_label")


def edge_cases(models, X):
    cases = {"all_missing": pd.DataFrame([{c: np.nan for c in X.columns}]), "extreme_x100": X.head(10) * 100,
             "extreme_negative": -X.head(10).abs() * 100, "single_row": X.head(1)}
    rows = []
    for name, m in models.items():
        for cname, d in cases.items():
            try:
                P = m.predict_proba(d)
                ok = bool(np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-5))
                msg = f"P range {P.min():.2f}-{P.max():.2f}"
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"error: {e}"
            rows.append({"Model": name, "case": cname, "passed": ok, "detail": msg})
        rows.append({"Model": name, "case": "deterministic", "passed": bool(np.allclose(m.predict_proba(X), m.predict_proba(X))),
                     "detail": ""})
    return pd.DataFrame(rows)


def summarise(ci, cvr, cor, adv, ln, perm, edge):
    rows = []
    for name in M:
        def add(t, v, ok, rule):
            rows.append({"Model": name, "Test": t, "Value": v, "Rule": rule, "Result": "PASS" if ok else "FAIL"})
        lo = ci[(ci.Model == name) & (ci.Metric == "Accuracy")].ci_low.iloc[0]
        add("1 accuracy CI lower bound", f"{lo:.3f}", lo > .61, "> 0.61 (always-Comedy)")
        s = cvr[cvr.Model == name]
        add("2 repeated CV accuracy (SD)", f"{s.Accuracy.mean():.3f} ({s.Accuracy.std():.3f})", s.Accuracy.std() <= .05, "SD <= 0.05")
        c = cor[cor.Model == name]
        base = c[c.test == "baseline"].Accuracy.iloc[0]
        g = lambda t, lv: c[(c.test == t) & np.isclose(c.level, lv)].Accuracy.iloc[0]  # noqa: E731
        add("3 noise 0.25 SD: accuracy drop", f"{base - g('noise_sd', .25):.3f}", base - g("noise_sd", .25) <= .05, "<= 0.05")
        add("4 +20% missing: accuracy drop", f"{base - g('missing', .2):.3f}", base - g("missing", .2) <= .05, "<= 0.05")
        raw, cl = g("unit_error_no_cleaning", .05), g("unit_error_with_cleaning", .05)
        add("5 A unit errors: drop raw / cleaned", f"{base - raw:.3f} / {base - cl:.3f}", base - cl <= .01, "cleaned <= 0.01")
        add("6 adversarial validation AUC", f"{adv:.3f}", adv <= .60, "<= 0.60")
        l = ln[ln.Model == name].iloc[0]
        add("7 10% label noise: accuracy drop", f"{l.clean - l.noisy:.3f}", l.clean - l.noisy <= .05, "<= 0.05")
        p = perm[perm.Model == name]
        add("8 permutation: real vs null macro-F1", f"{p.F1_real.iloc[0]:.3f} vs {p.F1_null.mean():.3f}",
            p.F1_real.iloc[0] - p.F1_null.max() >= .2, "gap >= 0.20")
        e = edge[edge.Model == name]
        add("11 edge cases", f"{e.passed.sum()}/{len(e)}", e.passed.all(), "all pass")
    return pd.DataFrame(rows)


def plot_all(ci, cvr, cor, ln, perm, lc, audit, summary, adv):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4))
    ax = axes[0, 0]
    mets = ["Accuracy", "Specificity_w", "F1_w", "F1_macro", "Sens_Comedy", "Sens_Drama", "Sens_Reality"]
    yy = np.arange(len(mets)); h = .26
    for i, m in enumerate(M):
        d = ci[ci.Model == m].set_index("Metric").loc[mets]
        ax.hlines(yy + (1 - i) * h, d.ci_low, d.ci_high, color=cfg.MODEL_COLORS[m], lw=2)
        ax.scatter(d["mean"], yy + (1 - i) * h, color=cfg.MODEL_COLORS[m], s=24, zorder=3, label=m)
    ax.axvline(.61, color=cfg.NEUTRAL, ls=":", lw=1); ax.axvline(cfg.TARGET_THRESHOLD, color="#e34948", ls="--", lw=1)
    ax.set_yticks(yy, mets); ax.invert_yaxis(); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("1. Bootstrap 95% CI on the 120-show holdout", fontsize=10); ax.grid(axis="y", visible=False)
    ax = axes[0, 1]
    data = [cvr[cvr.Model == m].Accuracy for m in M]
    bp = ax.boxplot(data, widths=.5, patch_artist=True, medianprops=dict(color=cfg.TEXT_PRIMARY))
    for p_, m in zip(bp["boxes"], M):
        p_.set_facecolor(cfg.MODEL_COLORS[m]); p_.set_alpha(.55)
    ax.set_xticks([1, 2, 3], M); ax.axhline(cfg.TARGET_THRESHOLD, color="#e34948", ls="--", lw=1)
    ax.set_ylabel("Fold accuracy"); ax.set_title("2. Repeated 5x3 CV on all 600 shows (red = 0.89 target)", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax = axes[0, 2]
    for m in M:
        d = cor[(cor.Model == m) & (cor.test == "noise_sd")]
        ax.plot(d.level, d.Accuracy, marker="o", ms=4, color=cfg.MODEL_COLORS[m], label=f"{m} noise")
        d = cor[(cor.Model == m) & (cor.test == "missing")]
        ax.plot(d.level, d.Accuracy, marker="s", ms=4, ls="--", color=cfg.MODEL_COLORS[m], label=f"{m} missing")
    ax.set_xlabel("Noise (x SD) or share of cells blanked"); ax.set_ylabel("Holdout accuracy")
    ax.legend(fontsize=6.5, ncol=2); ax.set_title("3-4. Noise (solid) and missing data (dashed)", fontsize=10)
    ax = axes[1, 0]
    x = np.arange(3); w = .26
    for j, (t, lab_) in enumerate([("baseline", "Clean"), ("unit_error_no_cleaning", "5% x100 errors, raw"),
                                   ("unit_error_with_cleaning", "5% x100 errors, cleaned")]):
        v = [cor[(cor.Model == m) & (cor.test == t)].Accuracy.iloc[0] for m in M]
        ax.bar(x + (j - 1) * w, v, w - .03, color=["#86b6ef", "#e34948", "#1baf7a"][j], label=lab_)
    ax.set_xticks(x, M); ax.set_ylim(.5, 1.0); ax.legend(fontsize=7.5, loc="lower right"); ax.grid(axis="x", visible=False)
    ax.set_title(f"5. Unit-error injection | 6. adversarial AUC = {adv:.2f}", fontsize=10)
    ax = axes[1, 1]
    for m in M:
        d = lc[lc.Model == m]
        ax.plot(d.n, d.Test_acc, marker="o", ms=4, color=cfg.MODEL_COLORS[m], label=f"{m} Test")
        ax.plot(d.n, d.Train_acc, ls="--", lw=1.3, color=cfg.MODEL_COLORS[m])
    ax.set_xlabel("Training shows"); ax.set_ylabel("Accuracy (dashed = Train)"); ax.legend(fontsize=7.5)
    ax.set_title("9. Learning curve", fontsize=10)
    ax = axes[1, 2]
    sus = audit[audit.suspect_label]
    tab = pd.crosstab(sus[cfg.TARGET], sus.oof_pred).reindex(index=cfg.CLASSES, columns=cfg.CLASSES, fill_value=0)
    ax.imshow(tab.values, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, tab.values[i, j], ha="center", va="center", fontsize=11,
                    color="white" if tab.values[i, j] > tab.values.max() / 2 else cfg.TEXT_PRIMARY)
    ax.set_xticks(range(3), [f"pred {c}" for c in cfg.CLASSES]); ax.set_yticks(range(3), [f"label {c}" for c in cfg.CLASSES])
    ax.grid(False); ax.set_title(f"10. Label audit: {len(sus)} confident disagreements (P >= 0.8)", fontsize=10)
    fig.suptitle("Stress tests (holdout = 120 shows unless stated)", x=.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "26_stress_tests")

    tests = summary.Test.unique()
    fig, ax = plt.subplots(figsize=(13, .45 * len(tests) + 1.3))
    ax.set_xlim(-.5, 3.9); ax.set_ylim(len(tests) - .5, -.5)
    for i, t in enumerate(tests):
        for j, m in enumerate(M):
            r = summary[(summary.Test == t) & (summary.Model == m)].iloc[0]
            ok = r.Result == "PASS"
            ax.add_patch(plt.Rectangle((j - .46, i - .4), .92, .8, color="#d6ecdf" if ok else "#f7d4d4", lw=0))
            ax.text(j, i, f"{r.Result}  {r.Value}", ha="center", va="center", fontsize=7.8,
                    color="#0b5d36" if ok else "#8f1d1d", fontweight="bold")
        ax.text(2.6, i, summary[summary.Test == t].Rule.iloc[0], va="center", fontsize=7.5, color=cfg.TEXT_SECONDARY)
    ax.set_xticks(range(3), M); ax.xaxis.tick_top(); ax.set_yticks(range(len(tests)), tests, fontsize=8.3)
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Stress-test scorecard (rules fixed before running)", pad=24)
    fig.tight_layout()
    save_fig(fig, "27_stress_test_scorecard")


def run() -> dict:
    cfg.ensure_dirs(); set_seed(cfg.RANDOM_STATE)
    feats = load_json(cfg.MODEL_METADATA_FILE)["features"]
    models = load_models("evaluation")
    Xtr, ytr = get_xy(load("train"), feats); X, y = get_xy(load("holdout"), feats)
    lab = load("labelled"); Xall, yall = get_xy(lab, feats)
    log.info("1 bootstrap"); ci = bootstrap(models, X, y)
    log.info("2 repeated CV"); cvr = repeated_cv(models, Xall, yall)
    log.info("3-5 corruption"); cor = corruption(models, Xtr, X, y, feats)
    log.info("6 adversarial"); adv = adversarial(lab, pd.read_csv(cfg.UNLABELLED_FILE), feats)
    log.info("7-9 retraining"); ln, perm, lc = retrain_tests(models, Xtr, ytr, X, y)
    log.info("10 label audit"); audit = label_audit(models, lab, feats)
    log.info("11 edge cases"); edge = edge_cases(models, X)
    for n, d in [("bootstrap_ci", ci), ("repeated_cv", cvr), ("corruption", cor), ("label_noise", ln),
                 ("permutation", perm), ("learning_curve", lc), ("label_audit", audit), ("edge_cases", edge)]:
        d.to_csv(cfg.TABLES_DIR / f"stress_{n}.csv", index=False)
    pd.DataFrame([{"adversarial_auc": adv}]).to_csv(cfg.TABLES_DIR / "stress_adversarial.csv", index=False)
    summary = summarise(ci, cvr, cor, adv, ln, perm, edge)
    summary.to_csv(cfg.TABLES_DIR / "stress_test_summary.csv", index=False)
    plot_all(ci, cvr, cor, ln, perm, lc, audit, summary, adv)
    log.info("Scorecard:\n%s", summary.pivot(index="Test", columns="Model", values="Result"))
    log.info("Repeated CV: %s", cvr.groupby("Model")[["Accuracy", "F1_macro", "AUC_macro"]].mean().round(3).to_dict())
    return {"summary": summary, "bootstrap": ci, "cv": cvr, "corruption": cor, "adversarial": adv,
            "label_noise": ln, "permutation": perm, "learning_curve": lc, "audit": audit, "edge": edge}


if __name__ == "__main__":
    run()

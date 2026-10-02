"""
Stage 4: optimise, tune and train the three genre classifiers.

Target: Accuracy, Sensitivity, Specificity, Precision and F1 (one-vs-rest,
support-weighted) >= 0.89 on Train and Test. Optimisation steps, each kept
only because it raised the cross-validated score (see figure 18):

    1. feature basis   the 5 independent columns C, I, L, M, N (stage 3)
    2. model shape     class-conditional Gaussian mixtures with 2 full-covariance
                       components per genre, matching the cluster structure
    3. semi-supervised the 400 shows of unknown genre are added to the EM fit
                       (their feature values only; no labels are used)
    4. stacking        XGBoost and the RBF-SVM learn on the mixture's clipped
                       log-posteriors (XGBoost also sees the 5 inputs)
    5. tuning          RandomizedSearchCV, 5-fold stratified, score = the WORST
                       of the five weighted target metrics, so the search
                       raises the metric that is furthest from the target

The three classifiers (one scikit-learn Pipeline each; blank inputs are
marginalised by the mixture instead of imputed):
    SSGMM     semi-supervised Gaussian-mixture classifier (24 candidates:
              components, covariance ridge, prior tempering, unlabelled on/off)
    XGBoost   mixture posteriors + inputs -> gradient-boosted trees (40)
    SVM-RBF   mixture posteriors -> z-score -> RBF-SVM, Platt probabilities (40)
A deep MLP on the same features was also optimised (12 candidates) and is
reported as a screened model: it ranked fourth on the tuning score, so it is
not deployed (only three classifiers are kept).

Overfitting guard: among candidates whose mean train-vs-validation score gap
is <= 0.05, the best validation score wins (else best validation, flagged).

Saved models:
    models/evaluation_<model>.joblib  fit on the 80% training split (Train / Test)
    models/production_<model>.joblib  same hyperparameters, refit on all 600
                                      labelled shows (used to classify the 400)

Run:  python -m src.models.train_model
"""
import platform
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src import config as cfg
from src.data.make_dataset import get_xy, load
from src.features.build_features import build_preprocessor
from src.models.gmm_models import GMMPosteriorFeatures, SemiSupervisedGMMClassifier
from src.models.metrics import MIN_WEIGHTED_SCORER
from src.models.mlp_model import DeepMLPClassifier
from src.utils import get_logger, load_json, save_json, set_seed
from src.visualization.visualize import save_fig

log = get_logger(__name__)
torch.set_num_threads(2)

SPACES = {
    "SSGMM": {
        "clf__n_components": [1, 2, 3], "clf__reg_covar": loguniform(1e-5, 1e-2),
        "clf__prior_power": uniform(0.25, 0.75), "clf__use_unlabelled": [True, False],
    },
    "XGBoost": {
        "gmm__n_components": [2, 3], "gmm__clip": [5.0, 10.0, 20.0], "gmm__components": [True, False],
        "clf__n_estimators": randint(100, 500), "clf__max_depth": randint(1, 5),
        "clf__learning_rate": loguniform(0.01, 0.2), "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 15),
        "clf__gamma": uniform(0, 3), "clf__reg_lambda": loguniform(1e-2, 20), "clf__reg_alpha": loguniform(1e-3, 5),
    },
    "SVM-RBF": {
        "gmm__n_components": [2, 3], "gmm__clip": [3.0, 5.0, 10.0], "gmm__components": [True, False],
        "clf__C": loguniform(1e-1, 3e1), "clf__gamma": loguniform(3e-3, 1), "clf__class_weight": [None, "balanced"],
    },
    "DeepMLP": {
        "gmm__clip": [5.0, 10.0], "clf__width": [32, 64, 128], "clf__n_layers": [1, 2, 3],
        "clf__dropout": uniform(0.0, 0.4), "clf__lr": loguniform(3e-4, 5e-3), "clf__weight_decay": loguniform(1e-5, 1e-2),
        "clf__batch_size": [32, 64], "clf__class_weight": [None, "balanced"],
    },
}


def unlabelled_matrix(feats):
    """Feature values of the 400 shows with unknown genre (used without labels by the mixtures)."""
    return pd.read_csv(cfg.UNLABELLED_FILE)[feats].values


def make_pipeline(name, U) -> Pipeline:
    """No imputer: the mixture marginalises over missing inputs and XGBoost routes NaN natively."""
    if name == "SSGMM":
        return Pipeline([("clf", SemiSupervisedGMMClassifier(2, X_unlabelled=U))])
    if name == "XGBoost":
        return Pipeline([("gmm", GMMPosteriorFeatures(2, X_unlabelled=U, keep_input=True)),
                         ("clf", XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                                               n_jobs=1, random_state=cfg.RANDOM_STATE))])
    if name == "SVM-RBF":
        return Pipeline([("gmm", GMMPosteriorFeatures(2, X_unlabelled=U)), ("scale", StandardScaler()),
                         ("clf", SVC(kernel="rbf", probability=False, random_state=cfg.RANDOM_STATE))])
    if name == "DeepMLP":
        return Pipeline([("gmm", GMMPosteriorFeatures(2, X_unlabelled=U)), ("scale", StandardScaler()),
                         ("clf", DeepMLPClassifier(max_epochs=200, patience=20, random_state=cfg.RANDOM_STATE))])
    raise ValueError(name)


def fit_params(name, y):
    """No sample weighting: the target metrics are support-weighted."""
    return {}


def pick_candidate(res, max_gap=cfg.MAX_TRAIN_VAL_GAP):
    """(index, guard_met): best validation score within the gap guard, else best overall."""
    gap = res.mean_train_score - res.mean_test_score
    ok = gap <= max_gap
    if ok.any():
        return int(res.loc[ok, "mean_test_score"].idxmax()), True
    return int(res["mean_test_score"].idxmax()), False


def tune(name, X, y, U):
    search = RandomizedSearchCV(make_pipeline(name, U), SPACES[name], n_iter=cfg.N_ITER[name],
                                scoring=MIN_WEIGHTED_SCORER, cv=StratifiedKFold(cfg.CV_FOLDS, shuffle=True,
                                                                       random_state=cfg.RANDOM_STATE),
                                refit=False, return_train_score=True, n_jobs=2, random_state=cfg.RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, **fit_params(name, y))
    res = pd.DataFrame(search.cv_results_)
    i, guard = pick_candidate(res)
    res["gap"] = res.mean_train_score - res.mean_test_score
    res["chosen"] = False; res.loc[i, "chosen"] = True
    keep = [c for c in res.columns if c.startswith("param_")] + ["mean_train_score", "mean_test_score",
                                                                 "std_test_score", "gap", "chosen"]
    res[keep].sort_values("mean_test_score", ascending=False).to_csv(
        cfg.TABLES_DIR / f"search_{name.lower().replace('-', '_')}.csv", index=False)
    log.info("%s: chosen CV score %.4f (train %.4f, gap %.3f, guard met: %s); best overall %.4f", name,
             res.loc[i, "mean_test_score"], res.loc[i, "mean_train_score"], res.loc[i, "gap"], guard,
             res.mean_test_score.max())
    res.attrs["guard_met"] = guard
    return res.loc[i, "params"], res


def fit_final(name, params, X, y, U):
    params = dict(params)
    if name == "SVM-RBF":
        params["clf__probability"] = True
    m = clone(make_pipeline(name, U)).set_params(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y, **fit_params(name, y))
    return m, params


def plot_search(results):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4))
    for ax, name in zip(axes, names):
        r = results[name]
        ax.scatter(r.mean_train_score, r.mean_test_score, s=30, alpha=.6, color=cfg.MODEL_COLORS[name],
                   edgecolor=cfg.SURFACE, label="Candidate")
        c = r[r.chosen]
        ax.scatter(c.mean_train_score, c.mean_test_score, s=140, marker="*", color=cfg.TEXT_PRIMARY, label="Chosen",
                   zorder=3)
        lo = min(r.mean_test_score.min(), r.mean_train_score.min())
        ax.plot([lo, 1], [lo, 1], color=cfg.NEUTRAL, ls="--", lw=1)
        ax.plot([lo, 1], [lo - cfg.MAX_TRAIN_VAL_GAP, 1 - cfg.MAX_TRAIN_VAL_GAP], color="#eb6834", ls=":", lw=1)
        ax.axhline(cfg.TARGET_THRESHOLD, color="#e34948", lw=.8, alpha=.6)
        ax.set_xlabel("Mean CV train score"); ax.set_ylabel("Mean CV validation score")
        tag = " (screened)" if name in cfg.SCREENED_MODELS else ""
        ax.set_title(f"{name}{tag}: {len(r)} candidates", fontsize=10); ax.legend(loc="lower right")
    fig.tight_layout()
    return save_fig(fig, "17_hyperparameter_search",
                    "Score = worst of the 5 weighted metrics. Dashed = no overfitting; dotted orange = 0.05 gap guard; "
                    "red line = 0.89 target")


def optimisation_ladder(Xtr, ytr, feats, results):
    """Repeated 5x2 CV on the training split for each optimisation step (cumulative)."""
    U = unlabelled_matrix(feats)
    sel = load_json(cfg.SELECTED_FEATURES_FILE)
    old = sel["strategies"]["S2 FDR + MI"]
    tr = load("train")
    yv = ytr.values
    steps = [
        ("0 always Comedy", lambda: DummyClassifier(strategy="most_frequent"), feats),
        ("1 previous release: RBF-SVM, 7 features", lambda: Pipeline([("prep", build_preprocessor()),
                                                                       ("clf", SVC(C=10, class_weight="balanced"))]), old),
        ("2 same SVM, 5-D latent basis", lambda: Pipeline([("prep", build_preprocessor()),
                                                           ("clf", SVC(C=10, class_weight="balanced"))]), feats),
        ("3 Gaussian mixture, 2 per genre", lambda: SemiSupervisedGMMClassifier(2, use_unlabelled=False), feats),
        ("4 + 400 unlabelled shows (semi-supervised)", lambda: SemiSupervisedGMMClassifier(2, X_unlabelled=U), feats),
    ]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=cfg.RANDOM_STATE)
    rows = []
    for label, mk, f in steps:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = cross_validate(mk(), tr[f], yv, cv=cv, scoring={"min": MIN_WEIGHTED_SCORER, "acc": "accuracy"})
        rows.append({"step": label, "score": r["test_min"].mean(), "sd": r["test_min"].std(),
                     "accuracy": r["test_acc"].mean(), "features": ",".join(f)})
    for name in cfg.MODEL_NAMES + cfg.SCREENED_MODELS:
        c = results[name][results[name].chosen].iloc[0]
        rows.append({"step": f"5 tuned {name}", "score": c.mean_test_score, "sd": c.std_test_score,
                     "accuracy": np.nan, "features": ",".join(feats)})
    return pd.DataFrame(rows)


def plot_ladder(lad):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    yy = np.arange(len(lad))[::-1]
    cols = [cfg.NEUTRAL] * 3 + ["#86b6ef", "#2a78d6"] + [
        cfg.MODEL_COLORS.get(s.replace("5 tuned ", ""), "#2a78d6") for s in lad.step[5:]]
    ax.barh(yy, lad.score, xerr=lad.sd, color=cols, height=.62, error_kw=dict(ecolor=cfg.TEXT_SECONDARY, lw=.8))
    for y_, v in zip(yy, lad.score):
        ax.text(max(v, 0) + .012, y_, f"{v:.3f}", va="center", fontsize=8.5)
    ax.axvline(cfg.TARGET_THRESHOLD, color="#e34948", ls="--", lw=1)
    ax.text(cfg.TARGET_THRESHOLD, yy[0] + .55, " 0.89 target", color="#b52f2f", fontsize=8)
    ax.set_yticks(yy, lad.step, fontsize=8.5); ax.set_xlim(0, 1.0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("CV score on the training split = worst of Accuracy, Sensitivity, Specificity, Precision, F1 "
                  "(weighted)", fontsize=8.5)
    ax.set_title("Optimisation steps (cumulative): what moved the weakest metric above 0.89", fontsize=10.5)
    fig.tight_layout()
    return save_fig(fig, "18_optimisation_steps")


def run() -> dict:
    cfg.ensure_dirs(); set_seed(cfg.RANDOM_STATE)
    feats = load_json(cfg.SELECTED_FEATURES_FILE)["selected_features"]
    U = unlabelled_matrix(feats)
    Xtr, ytr = get_xy(load("train"), feats)
    Xall, yall = get_xy(load("labelled"), feats)
    meta, results, evaluation = {}, {}, {}
    for name in cfg.MODEL_NAMES + cfg.SCREENED_MODELS:
        t0 = time.time()
        params, results[name] = tune(name, Xtr, ytr, U)
        ev, final_params = fit_final(name, params, Xtr, ytr, U)
        joblib.dump(ev, cfg.model_file(name, "evaluation"))
        if name in cfg.MODEL_NAMES:
            prod, _ = fit_final(name, params, Xall, yall, U)
            joblib.dump(prod, cfg.model_file(name, "production"))
        evaluation[name] = ev
        r = results[name][results[name].chosen].iloc[0]
        meta[name] = {"label": cfg.MODEL_LABELS.get(name, "Deep MLP on mixture posteriors (screened)"),
                      "deployed": name in cfg.MODEL_NAMES,
                      "best_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                      for k, v in final_params.items()},
                      "cv_score_min_weighted": float(r.mean_test_score), "cv_score_sd": float(r.std_test_score),
                      "cv_train_score": float(r.mean_train_score),
                      "gap_guard_met": bool(results[name].attrs["guard_met"]),
                      "evaluation_file": cfg.model_file(name, "evaluation").name,
                      "production_file": cfg.model_file(name, "production").name if name in cfg.MODEL_NAMES else None,
                      "train_seconds": round(time.time() - t0, 1)}
        log.info("%s trained in %.0fs", name, time.time() - t0)
    plot_search(results)
    lad = optimisation_ladder(Xtr, ytr, feats, results)
    lad.to_csv(cfg.TABLES_DIR / "optimisation_steps.csv", index=False); plot_ladder(lad)
    log.info("Optimisation steps:\n%s", lad[["step", "score", "accuracy"]].round(4).to_string(index=False))
    save_json({"features": feats, "classes": cfg.CLASSES, "tuning_score": "min of weighted Accuracy, Sensitivity, "
               "Specificity, Precision, F1", "n_unlabelled_used": int(len(U)), "models": meta,
               "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                               "xgboost": xgboost.__version__, "torch": torch.__version__}}, cfg.MODEL_METADATA_FILE)
    return {"models": evaluation, "metadata": meta, "search": results, "ladder": lad}


def load_models(stage="evaluation", include_screened=False) -> dict:
    names = cfg.MODEL_NAMES + (cfg.SCREENED_MODELS if include_screened and stage == "evaluation" else [])
    return {n: joblib.load(cfg.model_file(n, stage)) for n in names}


if __name__ == "__main__":
    run()

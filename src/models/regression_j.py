"""
Stage 8: regression model predicting Feature J from Feature C plus two
supporting features that are uncorrelated with C (and with each other).

Data: every show from BOTH files where J was actually observed (the 70 values
reconstructed during cleaning are excluded from fitting and used as an extra,
independent check). Shows are split 80 / 20 at random (Train / Test).

1. Supporting features (searched on the Train split; candidates: every
   cleaned feature except C and J). Two routes:
   a. As measured: pairs (X1, X2) where C, X1, X2 are mutually uncorrelated
      as recorded (|Pearson r| < 0.10 and p >= 0.05 for all three pairs).
   b. Orthogonalised (chosen): any pair, made uncorrelated by sequential
      residualisation (Gram-Schmidt, fitted on Train only):
          X1' = X1 - (a1 + b1 C)        X2' = X2 - (a2 + b2 C + c2 X1')
      so C, X1', X2' have zero correlation with each other (VIF = 1). X1'
      and X2' carry only the information in X1 and X2 that C does not.
   In each route the pair with the lowest 10-fold CV RMSE of OLS wins.

2. Model comparison on C + X1' + X2' (each tuned on the Train split by CV):
       OLS linear, Polynomial (degree by CV), Cubic spline + ridge,
       SVR-RBF (RandomizedSearchCV), XGBoost (RandomizedSearchCV)
   with the C-only OLS and the as-measured route as baselines. Metrics on
   Train and Test: MSE, RMSE, MAE (lower is better) and adjusted R^2 (higher
   is better; p = number of inputs). Best model: lowest 10-fold CV RMSE with
   the one-standard-error rule (the simplest model within 1 SE of the best).

3. Context (secondary): the C x genre interaction and the exact identity
   J = 0.446 B + 0.337 C + 0.643 I + 0.360 L - 0.282 M + 0.971 N.

Run:  python -m src.models.regression_j
"""
import itertools
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from src import config as cfg
from src.data.make_dataset import load
from src.utils import get_logger, save_json
from src.visualization.visualize import save_fig

log = get_logger(__name__)
REG_COLORS = {"OLS (C only)": cfg.NEUTRAL, "OLS (as measured)": "#b0aea8", "OLS linear": "#2a78d6",
              "Polynomial": "#eb6834", "Cubic spline": "#1baf7a", "SVR-RBF": "#e87ba4", "XGBoost": "#4a3aa7"}
MODEL_FILE = cfg.MODELS_DIR / "regression_j.joblib"
MAX_ABS_R, MIN_P = 0.10, 0.05
CV = KFold(10, shuffle=True, random_state=cfg.RANDOM_STATE)
COMPLEXITY = ["OLS linear", "Polynomial", "SVR-RBF", "Cubic spline", "XGBoost"]   # simplest first


class Orthogonalizer(TransformerMixin, BaseEstimator):
    """Sequential residualisation: column j becomes its residual on columns 0..j-1 (column 0 unchanged).

    Fitted on the training rows only; on those rows the output columns are mutually uncorrelated.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.coefs_ = []
        Z = X[:, :1].copy()
        for j in range(1, X.shape[1]):
            A = np.column_stack([np.ones(len(X)), Z])
            beta = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
            self.coefs_.append(beta)
            Z = np.column_stack([Z, X[:, j] - A @ beta])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = X[:, :1].copy()
        for j, beta in enumerate(self.coefs_, start=1):
            Z = np.column_stack([Z, X[:, j] - np.column_stack([np.ones(len(X)), Z]) @ beta])
        return Z


def ortho(*steps):
    return make_pipeline(Orthogonalizer(), *steps)


def regression_data():
    lab, unl = load("labelled"), pd.read_csv(cfg.UNLABELLED_FILE)
    both = pd.concat([lab.assign(source="training file"), unl.assign(source="test file")], ignore_index=True)
    observed = both[both.J_was_reconstructed == 0].reset_index(drop=True)
    reconstructed = both[both.J_was_reconstructed == 1].reset_index(drop=True)
    return observed, reconstructed, lab


def scores(m, X, y):
    p = m.predict(X); n, k = X.shape
    r2 = r2_score(y, p); mse = mean_squared_error(y, p)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y, p), "R2": r2,
            "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - k - 1)}


def cv_rmse_folds(m, X, y):
    return -cross_val_score(clone(m), X, y, cv=CV, scoring="neg_root_mean_squared_error")


def screen_support_features(tr):
    """Every candidate pair: correlations with C and each other, and CV RMSE of OLS J ~ C + X1 + X2."""
    cands = [c for c in cfg.CLEAN_FEATURES if c not in ("C", "J")]
    rows = []
    for a, b in itertools.combinations(cands, 2):
        rs = [stats.pearsonr(tr[u], tr[v]) for u, v in [("C", a), ("C", b), (a, b)]]
        rows.append({"X1": a, "X2": b, "r_C_X1": rs[0].statistic, "r_C_X2": rs[1].statistic,
                     "r_X1_X2": rs[2].statistic, "max_abs_r": max(abs(r.statistic) for r in rs),
                     "min_p": min(r.pvalue for r in rs),
                     "uncorrelated_as_measured": all(abs(r.statistic) < MAX_ABS_R and r.pvalue >= MIN_P for r in rs),
                     "cv_rmse_ols": cv_rmse_folds(LinearRegression(), tr[["C", a, b]], tr["J"]).mean()})
    t = pd.DataFrame(rows).sort_values("cv_rmse_ols").reset_index(drop=True)
    raw = t[t.uncorrelated_as_measured].iloc[0]
    best = t.iloc[0]          # after orthogonalisation every pair is uncorrelated with C and with each other
    return t, [raw.X1, raw.X2], [best.X1, best.X2]


def fit_models(Xtr, ytr):
    models, info = {}, {}
    models["OLS linear"] = ortho(LinearRegression()).fit(Xtr, ytr)
    deg = {d: cv_rmse_folds(ortho(PolynomialFeatures(d), LinearRegression()), Xtr, ytr).mean() for d in range(1, 5)}
    info["poly_cv_rmse_by_degree"] = deg; info["poly_degree"] = min(deg, key=deg.get)
    models["Polynomial"] = ortho(PolynomialFeatures(info["poly_degree"]), LinearRegression()).fit(Xtr, ytr)
    kn = {k: cv_rmse_folds(ortho(SplineTransformer(n_knots=k, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))),
                           Xtr, ytr).mean() for k in [3, 4, 5, 6, 8]}
    info["spline_cv_rmse_by_knots"] = kn; info["spline_knots"] = min(kn, key=kn.get)
    models["Cubic spline"] = ortho(SplineTransformer(n_knots=info["spline_knots"], degree=3),
                                   RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(Xtr, ytr)
    svr = RandomizedSearchCV(ortho(StandardScaler(), SVR()),
                             {"svr__C": loguniform(.1, 100), "svr__gamma": loguniform(1e-3, 1),
                              "svr__epsilon": loguniform(1e-2, 1)},
                             n_iter=30, cv=CV, scoring="neg_root_mean_squared_error", random_state=cfg.RANDOM_STATE,
                             n_jobs=2).fit(Xtr, ytr)
    models["SVR-RBF"] = svr.best_estimator_
    info["svr_params"] = {k: float(v) for k, v in svr.best_params_.items()}
    xgb = RandomizedSearchCV(ortho(XGBRegressor(n_jobs=1, random_state=cfg.RANDOM_STATE, verbosity=0)),
                             {f"xgbregressor__{k}": v for k, v in {
                                 "n_estimators": randint(50, 600), "max_depth": randint(1, 5),
                                 "learning_rate": loguniform(.01, .2), "min_child_weight": randint(5, 60),
                                 "subsample": uniform(.6, .4), "reg_lambda": loguniform(.1, 50)}.items()},
                             n_iter=30, cv=CV, scoring="neg_root_mean_squared_error", random_state=cfg.RANDOM_STATE,
                             n_jobs=2).fit(Xtr, ytr)
    models["XGBoost"] = xgb.best_estimator_
    info["xgb_params"] = {k.replace("xgbregressor__", ""): float(v) for k, v in xgb.best_params_.items()}
    return models, info


def run() -> dict:
    cfg.ensure_dirs()
    obs, rec, lab = regression_data()
    tr, te = train_test_split(obs, test_size=.2, random_state=cfg.RANDOM_STATE)
    screen, raw_support, support = screen_support_features(tr)
    screen.round(5).to_csv(cfg.TABLES_DIR / "regression_j_support_feature_screening.csv", index=False)
    feats, rawf = ["C"] + support, ["C"] + raw_support
    log.info("Uncorrelated as measured: %s | orthogonalised pair (chosen): %s", raw_support, support)
    Xtr, ytr, Xte, yte = tr[feats], tr["J"], te[feats], te["J"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models, info = fit_models(Xtr, ytr)
    baselines = {"OLS (C only)": (LinearRegression().fit(tr[["C"]], ytr), ["C"]),
                 "OLS (as measured)": (LinearRegression().fit(tr[rawf], ytr), rawf)}
    rows = []
    for name, (m, fs) in list(baselines.items()) + [(n, (m, feats)) for n, m in models.items()]:
        label = " + ".join(fs) if name in baselines else " + ".join(["C"] + [f"{f}'" for f in support])
        cvs = cv_rmse_folds(m, tr[fs], ytr)
        rows.append({"Model": name, "Features": label,
                     **{f"Train_{k}": v for k, v in scores(m, tr[fs], ytr).items()},
                     **{f"Test_{k}": v for k, v in scores(m, te[fs], yte).items()},
                     "CV10_RMSE": cvs.mean(), "CV10_RMSE_SE": cvs.std(ddof=1) / np.sqrt(len(cvs)),
                     **{f"Reconstructed_{k}": v for k, v in scores(m, rec[fs], rec["J"]).items()}})
    comp = pd.DataFrame(rows)
    comp.round(5).to_csv(cfg.TABLES_DIR / "regression_j_model_comparison.csv", index=False)
    cand = comp[comp.Model.isin(COMPLEXITY)].set_index("Model")
    lowest = cand.CV10_RMSE.idxmin()
    limit = cand.loc[lowest, "CV10_RMSE"] + cand.loc[lowest, "CV10_RMSE_SE"]
    best = next(m for m in COMPLEXITY if cand.loc[m, "CV10_RMSE"] <= limit)
    info["lowest_cv_rmse_model"] = lowest; info["one_se_limit"] = float(limit)

    # correlation of the predictors before / after orthogonalisation (Train and Test)
    orth = models["OLS linear"].named_steps["orthogonalizer"]
    onames = ["C"] + [f"{f}'" for f in support]
    corr = pd.DataFrame([{"split": sp, "max_abs_r_as_measured": float(np.abs(d[feats].corr().values - np.eye(3)).max()),
                          "max_abs_r_orthogonalised": float(np.abs(np.corrcoef(orth.transform(d[feats]).T)
                                                                   - np.eye(3)).max())}
                         for sp, d in [("Train", tr), ("Test", te)]])
    corr.to_csv(cfg.TABLES_DIR / "regression_j_predictor_correlation.csv", index=False)

    # OLS inference + diagnostics on the orthogonalised predictors; VIF as measured vs orthogonalised
    Z = pd.DataFrame(orth.transform(Xtr), columns=feats, index=tr.index).assign(J=ytr.values)
    formula = "J ~ " + " + ".join(feats)
    ols = smf.ols(formula, data=Z).fit()
    robust = smf.ols(formula, data=Z).fit(cov_type="HC3")
    raw_ols = smf.ols(formula, data=tr).fit()
    bp = het_breuschpagan(ols.resid, ols.model.exog)
    vif = {f: float(variance_inflation_factor(ols.model.exog, i + 1)) for i, f in enumerate(feats)}
    vif_raw = {f: float(variance_inflation_factor(raw_ols.model.exog, i + 1)) for i, f in enumerate(feats)}
    ols_tab = pd.DataFrame({"coef": ols.params, "se": ols.bse, "se_HC3": robust.bse, "p": ols.pvalues,
                            "ci_low": ols.conf_int()[0], "ci_high": ols.conf_int()[1],
                            "VIF_orthogonalised": pd.Series(vif), "VIF_as_measured": pd.Series(vif_raw),
                            "coef_as_measured": raw_ols.params})
    ols_tab.index = [i if i in ("Intercept", "C") else f"{i}'" for i in ols_tab.index]
    ols_tab.to_csv(cfg.TABLES_DIR / "regression_j_ols_coefficients.csv")

    # Secondary: genre interaction and the exact identity
    gl = lab[lab.J_was_reconstructed == 0]
    inter = smf.ols("J ~ C * genre", data=gl).fit()      # genre is text, so patsy treats it as categorical
    slopes = {g: smf.ols("J ~ C", data=gl[gl.genre == g]).fit() for g in cfg.CLASSES}
    exact = LinearRegression().fit(tr[cfg.J_SOURCES], ytr)
    secondary = pd.DataFrame([
        {"model": "J ~ C (OLS, training file)", "R2": smf.ols("J ~ C", data=gl).fit().rsquared, "n": len(gl)},
        {"model": "J ~ C x genre (training file)", "R2": inter.rsquared, "n": len(gl)},
        *[{"model": f"J ~ C within {g}", "R2": s_.rsquared, "n": int(s_.nobs), "slope": s_.params["C"]}
          for g, s_ in slopes.items()],
        {"model": "J ~ B + C + I + L + M + N (exact identity, Test)", "R2": exact.score(te[cfg.J_SOURCES], yte),
         "n": len(te)}])
    secondary.round(5).to_csv(cfg.TABLES_DIR / "regression_j_secondary.csv", index=False)

    joblib.dump({"model": models[best], "name": best, "features": feats, "target": "J",
                 "note": "the pipeline orthogonalises inputs 2 and 3 against the preceding inputs"}, MODEL_FILE)
    joblib.dump(models["OLS linear"], cfg.MODELS_DIR / "regression_j_ols.joblib")
    save_json({"best_model": best, "features": feats, "support_features": support,
               "uncorrelated_as_measured_pair": raw_support,
               "rule": f"as measured: |r| < {MAX_ABS_R} and p >= {MIN_P} for C-X1, C-X2, X1-X2; orthogonalised: "
                       "sequential residualisation fitted on the Train split",
               "selection": "lowest 10-fold CV RMSE, one-standard-error rule (simplest model within 1 SE)",
               "ols_orthogonalised": {k: float(v) for k, v in ols.params.items()},
               "ols_r2": float(ols.rsquared), "ols_adj_r2": float(ols.rsquared_adj), "vif_orthogonalised": vif,
               "vif_as_measured": vif_raw, "breusch_pagan_p": float(bp[1]),
               "predictor_correlation": corr.to_dict("records"),
               **{k: ({str(a): float(b) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in info.items()},
               "n_observed": len(obs), "n_train": len(tr), "n_test": len(te), "n_reconstructed_check": len(rec)},
              cfg.MODELS_DIR / "regression_j_metadata.json")

    plot_screening(tr, screen, raw_support, support, orth)
    plot_diagnostics(ols, bp, feats)
    plot_genre(gl, slopes, inter)
    plot_comparison(comp, models, best, Xte, yte)
    log.info("Regression comparison:\n%s", comp[["Model", "Features", "Train_RMSE", "Test_RMSE", "Test_MSE",
                                                 "Test_MAE", "Train_Adj_R2", "Test_Adj_R2", "CV10_RMSE"]].round(4))
    log.info("Best (1-SE rule): %s (lowest CV RMSE: %s) | VIF as measured %s -> %s | correlation:\n%s", best, lowest,
             {k: round(v, 2) for k, v in vif_raw.items()}, {k: round(v, 2) for k, v in vif.items()}, corr.round(4))
    return {"comparison": comp, "ols": ols, "ols_table": ols_tab, "secondary": secondary, "models": models,
            "best": best, "info": info, "interaction": inter, "breusch_pagan": bp, "screening": screen,
            "features": feats, "support": support, "raw_support": raw_support, "correlation": corr,
            "orthogonalizer": orth, "onames": onames}


# --------------------------------------------------------------------------- #
def plot_screening(tr, screen, raw_support, support, orth):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [.75, .75, 1.4]})
    feats = ["C"] + support
    Z = pd.DataFrame(orth.transform(tr[feats]), columns=["C"] + [f"{f}'" for f in support])
    for ax, R, title in [(axes[0], tr[feats].corr(), f"As measured: {support[0]} and {support[1]} correlate with C"),
                         (axes[1], Z.corr(), "After orthogonalisation: r = 0, VIF = 1")]:
        ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(3):
            for j in range(3):
                v = R.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(v) > .6 else cfg.TEXT_PRIMARY)
        ax.set_xticks(range(3), R.columns); ax.set_yticks(range(3), R.columns); ax.grid(False)
        ax.set_title(title, fontsize=9.5)
    top = pd.concat([screen.head(10), screen[screen.uncorrelated_as_measured].head(4)]).drop_duplicates(["X1", "X2"])
    top = top.sort_values("cv_rmse_ols", ascending=False)
    col = ["#1c5cab" if (a, b) == tuple(support) else ("#1baf7a" if e else "#86b6ef")
           for a, b, e in zip(top.X1, top.X2, top.uncorrelated_as_measured)]
    ax = axes[2]
    ax.barh([f"C + {a} + {b}  (max |r| as measured {m:.2f})" for a, b, m in zip(top.X1, top.X2, top.max_abs_r)],
            top.cv_rmse_ols, color=col, height=.65)
    for y_, v in enumerate(top.cv_rmse_ols):
        ax.text(v + .01, y_, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlabel("10-fold CV RMSE of OLS J ~ C + X1 + X2 on the Train split (same before and after orthogonalising)",
                  fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5); ax.grid(axis="y", visible=False)
    ax.set_title(f"Pair search: dark blue = chosen ({support[0]}, {support[1]}); green = uncorrelated as measured "
                 f"(best: {raw_support[0]}, {raw_support[1]})", fontsize=8.5)
    fig.tight_layout()
    return save_fig(fig, "29_regression_J_support_feature_screening")


def plot_diagnostics(ols, bp, feats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    f, r = ols.fittedvalues, ols.resid
    axes[0].scatter(f, r, s=9, alpha=.5, color="#2a78d6"); axes[0].axhline(0, color=cfg.NEUTRAL, lw=1)
    axes[0].set_xlabel("Fitted J"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residuals vs fitted", fontsize=10)
    (osm, osr), (sl, ic, _) = stats.probplot(r, dist="norm")
    axes[1].scatter(osm, osr, s=9, color="#2a78d6"); axes[1].plot(osm, sl * np.asarray(osm) + ic, color="#e34948", lw=1.2)
    axes[1].set_xlabel("Theoretical quantile"); axes[1].set_ylabel("Residual quantile")
    axes[1].set_title(f"Normal Q-Q (Shapiro p = {stats.shapiro(r).pvalue:.2g})", fontsize=10)
    axes[2].scatter(f, np.sqrt(np.abs(r / r.std())), s=9, alpha=.5, color="#2a78d6")
    axes[2].set_xlabel("Fitted J"); axes[2].set_ylabel("sqrt(|standardised residual|)")
    axes[2].set_title(f"Scale-location (Breusch-Pagan p = {bp[1]:.2g})", fontsize=10)
    eq = " ".join(f"{ols.params[c]:+.3f}·{c if c == 'C' else c + chr(39)}" for c in feats)
    fig.suptitle(f"OLS diagnostics (Train): J = {ols.params['Intercept']:.3f} {eq}   adj. R^2 = {ols.rsquared_adj:.3f}",
                 x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "30_regression_J_ols_diagnostics")


def plot_genre(gl, slopes, inter):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    grid = np.linspace(gl.C.min(), gl.C.max(), 100)
    for g in cfg.CLASSES:
        d = gl[gl.genre == g]; s = slopes[g]
        ax.scatter(d.C, d.J, s=11, alpha=.45, color=cfg.COLORS[g], edgecolor="none")
        ax.plot(grid, s.params["Intercept"] + s.params["C"] * grid, color=cfg.COLORS[g], lw=2.2,
                label=f"{g}: slope {s.params['C']:+.2f}, R^2 {s.rsquared:.2f}")
    ax.set_xlabel("Feature C"); ax.set_ylabel("Feature J"); ax.legend(fontsize=8)
    ax.set_title(f"Secondary: C alone explains little of J (J ~ C x genre R^2 = {inter.rsquared:.2f})", fontsize=10.5)
    fig.tight_layout()
    return save_fig(fig, "31_regression_J_vs_C_by_genre")


def plot_comparison(comp, models, best, Xte, yte):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    x = np.arange(len(comp)); w = .38
    for ax, met, lab in zip(axes[:3], ["RMSE", "MAE", "Adj_R2"],
                            ["RMSE (lower is better)", "MAE (lower is better)", "Adjusted R^2 (higher is better)"]):
        ax.bar(x - w / 2, comp[f"Train_{met}"], w - .03, color="#86b6ef", label="Train")
        ax.bar(x + w / 2, comp[f"Test_{met}"], w - .03, color="#1c5cab", label="Test")
        for xx, v in zip(x + w / 2, comp[f"Test_{met}"]):
            ax.text(xx, max(v, 0), f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x, [m.replace(" (", "\n(") for m in comp.Model], fontsize=7, rotation=35)
        ax.set_title(lab, fontsize=10); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8)
    p = models[best].predict(Xte)
    axes[3].scatter(yte, p, s=12, alpha=.6, color=REG_COLORS[best])
    lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]
    axes[3].plot(lim, lim, color=cfg.NEUTRAL, ls="--", lw=1)
    axes[3].set_xlabel("Observed J (Test)"); axes[3].set_ylabel(f"Predicted J ({best})")
    axes[3].set_title(f"Best: {best} (Test RMSE {np.sqrt(mean_squared_error(yte, p)):.3f})", fontsize=10)
    fig.suptitle(f"J regression on {comp.Features.iloc[-1]} (orthogonalised) vs the C-only and as-measured baselines",
                 x=.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "32_regression_J_model_comparison")


if __name__ == "__main__":
    run()

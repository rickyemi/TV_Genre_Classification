"""
Gaussian-mixture models for the genre classifiers.

Why Gaussian mixtures: the audit showed that the informative features span a
5-dimensional space (C, I, L, M, N; the other informative columns are exact or
near-exact linear combinations of these) and that each genre is made of about
two elongated Gaussian clusters with their own covariance. A class-conditional
mixture of full-covariance Gaussians is the natural (near Bayes-optimal) model
for that shape, which axis-aligned trees and an isotropic RBF kernel struggle
to approximate from 480 shows.

Classes
-------
SemiSupervisedGMMClassifier
    k Gaussian components per genre. Fitted by EM on the labelled shows (each
    labelled show may only belong to components of its own genre) PLUS the
    400 shows of unknown genre (free to belong to any component). The
    unlabelled shows sharpen the cluster shapes; their labels are never used,
    because they have none. Posterior P(genre | x) = sum of that genre's
    component densities x priors, normalised. `prior_power` tempers the class
    priors (1 = empirical frequencies, 0 = uniform) and is tuned by CV.

GMMPosteriorFeatures
    scikit-learn transformer that fits the semi-supervised mixture on the
    training rows and outputs clipped log-posteriors for every genre and every
    component (optionally next to the input features). XGBoost and the RBF-SVM
    are trained on these features: the mixture describes the cluster geometry,
    the discriminative model learns the decision boundary on top of it.

Missing inputs are handled exactly rather than imputed: a Gaussian's marginal
over the observed features is again Gaussian, so a show with blank features is
scored on the features it has (a show with no features gets the class priors).
Median imputation would place such a show off the thin cluster shapes.

All estimators follow the scikit-learn API (get_params / set_params / clone),
so they are tuned with RandomizedSearchCV, stress-tested and saved with joblib
like any other model.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.mixture import GaussianMixture


def _log_gauss_complete(X, mu, S):
    d = X.shape[1]
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (np.sum(z ** 2, axis=0) + d * np.log(2 * np.pi)) - np.sum(np.log(np.diag(L)))


def _log_gauss(X, mu, S):
    """log N(x_obs | mu_obs, S_obs) for every row of X, marginalising over missing (NaN) entries."""
    miss = np.isnan(X)
    if not miss.any():
        return _log_gauss_complete(X, mu, S)
    out = np.zeros(len(X))
    patterns, inv = np.unique(miss, axis=0, return_inverse=True)
    for k, pat in enumerate(patterns):
        rows = np.where(inv.ravel() == k)[0]
        o = ~pat
        if o.any():
            out[rows] = _log_gauss_complete(X[np.ix_(rows, o)], mu[o], S[np.ix_(o, o)])
    return out


class SemiSupervisedGMMClassifier(ClassifierMixin, BaseEstimator):
    """Class-conditional Gaussian mixture fitted by semi-supervised EM.

    Parameters
    ----------
    n_components : components per genre
    X_unlabelled : array (n, d) of shows with unknown genre (same columns as X), or None
    use_unlabelled : if False the model is a purely supervised mixture discriminant
    reg_covar : ridge added to every covariance diagonal
    prior_power : class prior = empirical frequency ** prior_power (renormalised)
    """

    def __init__(self, n_components=2, X_unlabelled=None, use_unlabelled=True, reg_covar=1e-4,
                 prior_power=1.0, max_iter=200, tol=1e-7, n_init=5, random_state=0):
        self.n_components = n_components
        self.X_unlabelled = X_unlabelled
        self.use_unlabelled = use_unlabelled
        self.reg_covar = reg_covar
        self.prior_power = prior_power
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        if np.isnan(X).any():                       # training rows are expected complete; fill any gaps
            X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        self.classes_ = np.unique(y)
        C, k, d = len(self.classes_), int(self.n_components), X.shape[1]
        yi = np.searchsorted(self.classes_, y)
        # 1. supervised start: one GaussianMixture per genre
        mu, S, w = [], [], []
        for c in range(C):
            Xc = X[yi == c]
            kc = min(k, len(Xc))
            g = GaussianMixture(kc, covariance_type="full", reg_covar=self.reg_covar, n_init=self.n_init,
                                random_state=self.random_state).fit(Xc)
            for j in range(k):
                jj = j % kc
                mu.append(g.means_[jj]); S.append(g.covariances_[jj])
                w.append(np.mean(yi == c) * g.weights_[jj])
        mu, S, w = np.array(mu), np.array(S), np.array(w) / np.sum(w)
        comp_class = np.repeat(np.arange(C), k)
        # 2. semi-supervised EM over labelled + unlabelled shows
        U = (np.asarray(self.X_unlabelled, dtype=float)
             if (self.use_unlabelled and self.X_unlabelled is not None) else np.empty((0, d)))
        A = np.vstack([X, U])
        allowed = np.ones((len(A), C * k), dtype=bool)
        allowed[:len(X)] = comp_class[None, :] == yi[:, None]
        prev = -np.inf
        for it in range(self.max_iter):
            L = np.column_stack([np.log(w[j] + 1e-300) + _log_gauss(A, mu[j], S[j]) for j in range(C * k)])
            L = np.where(allowed, L, -np.inf)
            m = L.max(1, keepdims=True)
            R = np.exp(L - m); tot = R.sum(1, keepdims=True)
            ll = float(np.sum(np.log(tot[:, 0]) + m[:, 0]))
            R /= tot
            Nk = R.sum(0) + 1e-10
            w = Nk / Nk.sum()
            mu = (R.T @ A) / Nk[:, None]
            S = np.array([((R[:, j, None] * (A - mu[j])).T @ (A - mu[j])) / Nk[j] + self.reg_covar * np.eye(d)
                          for j in range(C * k)])
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        self.means_, self.covariances_, self.weights_, self.component_class_ = mu, S, w, comp_class
        self.n_iter_, self.log_likelihood_ = it + 1, ll
        # class priors: labelled frequencies, tempered
        freq = np.bincount(yi, minlength=C) / len(yi)
        p = freq ** self.prior_power; p /= p.sum()
        mass = np.array([w[comp_class == c].sum() for c in range(C)])
        self.log_prior_adjust_ = np.log(p) - np.log(mass)   # replace mixture class mass by the chosen prior
        return self

    # -------------------------------------------------------------- predict
    def component_log_density(self, X):
        """log w_j + log N(x | mu_j, S_j) for every component j (columns)."""
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.log(self.weights_[j]) + _log_gauss(X, self.means_[j], self.covariances_[j])
                                for j in range(len(self.weights_))])

    def class_log_score(self, X):
        Lj = self.component_log_density(X)
        out = np.column_stack([np.logaddexp.reduce(Lj[:, self.component_class_ == c], axis=1)
                               for c in range(len(self.classes_))])
        return out + self.log_prior_adjust_, Lj

    def predict_proba(self, X):
        L, _ = self.class_log_score(X)
        L = L - L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class GMMPosteriorFeatures(TransformerMixin, BaseEstimator):
    """Clipped log-posterior features from a semi-supervised class-conditional mixture.

    Output columns: [input features if keep_input] + one per genre
    (log P(genre | x) - max) + one per component if `components` (log
    responsibility - max). Values are clipped to [-clip, 0] so a single
    far-away show cannot produce extreme feature values.
    """

    def __init__(self, n_components=2, X_unlabelled=None, use_unlabelled=True, reg_covar=1e-4,
                 clip=10.0, components=True, keep_input=False, random_state=0):
        self.n_components = n_components
        self.X_unlabelled = X_unlabelled
        self.use_unlabelled = use_unlabelled
        self.reg_covar = reg_covar
        self.clip = clip
        self.components = components
        self.keep_input = keep_input
        self.random_state = random_state

    def fit(self, X, y):
        self.gmm_ = SemiSupervisedGMMClassifier(self.n_components, self.X_unlabelled, self.use_unlabelled,
                                                self.reg_covar, random_state=self.random_state).fit(X, y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Lc, Lj = self.gmm_.class_log_score(X)
        F = [Lc - Lc.max(1, keepdims=True)]
        if self.components:
            F.append(Lj - Lj.max(1, keepdims=True))
        F = np.clip(np.column_stack(F), -self.clip, 0.0)
        return np.column_stack([X, F]) if self.keep_input else F

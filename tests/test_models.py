"""Feature selection, metrics, Gaussian-mixture models, saved models, predictions and the J regression."""
import numpy as np
import pandas as pd

from src import config as cfg
from src.data.make_dataset import get_xy
from src.features.build_features import IQRCapper, vif
from src.models.evaluate_model import multiclass_metrics
from src.models.mlp_model import DeepMLPClassifier
from src.utils import load_json


def test_iqr_capper_and_vif():
    X = np.array([[1.], [2.], [3.], [4.], [100.]])
    assert IQRCapper().fit(X).transform(X)[4, 0] < 100
    df = pd.DataFrame({"a": [1., 2, 3, 4, 5], "b": [2., 1, 4, 3, 6]}); df["c"] = df.a + df.b
    assert np.isinf(vif(df)["c"])


def test_multiclass_metrics_known_case():
    y = np.array([0, 0, 1, 1, 2, 2])
    P = np.eye(3)[[0, 0, 1, 2, 2, 2]] * .9 + .1 / 3
    m = multiclass_metrics(y, P)
    assert np.isclose(m["Accuracy"], 5 / 6)
    assert np.isclose(m["per_class"]["Drama"]["Sensitivity"], .5)
    assert np.isclose(m["per_class"]["Reality"]["Precision"], 2 / 3)
    # weighted sensitivity equals accuracy; weighted precision uses genre shares as weights
    assert np.isclose(m["Sensitivity"], m["Accuracy"])
    assert np.isclose(m["Precision"], (1 + 1 + 2 / 3) / 3)


def test_semi_supervised_gmm_recovers_clusters():
    from src.models.gmm_models import GMMPosteriorFeatures, SemiSupervisedGMMClassifier
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [4, 0], [0, 4], [4, 4]]); lab = np.array([0, 0, 1, 2])
    z = rng.integers(0, 4, 600); X = centres[z] + rng.normal(size=(600, 2)) * .7; y = lab[z]
    clf = SemiSupervisedGMMClassifier(2, X_unlabelled=X[400:]).fit(X[:400], y[:400])
    P = clf.predict_proba(X[400:])
    assert np.allclose(P.sum(1), 1) and (P.argmax(1) == y[400:]).mean() > .9
    F = GMMPosteriorFeatures(2, X_unlabelled=X[400:], clip=5).fit(X[:400], y[:400]).transform(X[400:])
    assert F.shape == (200, 3 + 6) and F.max() <= 0 and F.min() >= -5


def test_mlp_learns_simple_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    clf = DeepMLPClassifier(max_epochs=80, patience=15).fit(X, y)
    assert (clf.predict(X) == y).mean() > .8


def test_saved_models_meet_regression_floor(models, data):
    feats = load_json(cfg.MODEL_METADATA_FILE)["features"]
    X, y = get_xy(data["holdout"], feats)
    for name, m in models["evaluation"].items():
        P = m.predict_proba(X)
        assert np.allclose(P.sum(1), 1, atol=1e-5)
        assert (P.argmax(1) == y).mean() > .65, name     # well above the 0.61 always-Comedy rate


def test_predictions_file_contract():
    p = pd.read_csv(cfg.PREDICTIONS_FILE)
    assert len(p) == 400 and set(p.Predicted_Genre) <= set(cfg.CLASSES)
    for name in cfg.MODEL_NAMES:
        probs = p[[f"{name.replace('-', '_')}_P_{c}" for c in cfg.CLASSES]].sum(1)
        assert np.allclose(probs, 1, atol=1e-3)
    assert p.Confidence.between(1 / 3, 1).all()


def test_orthogonalizer_decorrelates():
    from src.models.regression_j import Orthogonalizer
    rng = np.random.default_rng(2)
    a = rng.normal(size=500); X = np.c_[a, a + rng.normal(size=500) * .3, a - rng.normal(size=500)]
    Z = Orthogonalizer().fit(X).transform(X)
    assert np.allclose(Z[:, 0], X[:, 0])
    assert np.abs(np.corrcoef(Z.T) - np.eye(3)).max() < 1e-8


def test_regression_j_model_loads_and_predicts():
    import joblib
    b = joblib.load(cfg.MODELS_DIR / "regression_j.joblib")
    assert b["features"][0] == "C" and len(b["features"]) == 3
    X = pd.DataFrame(np.zeros((3, 3)), columns=b["features"]); X["C"] = [-1.0, 0.0, 1.0]
    pred = b["model"].predict(X)
    assert np.isfinite(pred).all() and len(pred) == 3

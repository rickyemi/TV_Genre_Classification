"""Raw data contract and the cleaning rules found in the audit."""
import numpy as np
import pandas as pd

from src import config as cfg
from src.data.make_dataset import clean, fit_cleaning_spec, load_raw


def test_raw_shapes_and_labels():
    tr, te = load_raw()
    assert tr.shape == (600, 16) and te.shape == (400, 16)
    assert set(tr[cfg.TARGET]) == set(cfg.CLASSES) and te[cfg.TARGET].isna().all()


def test_duplicate_columns_are_exact_copies_of_I():
    tr, _ = load_raw()
    for col in ["D", "F", "K"]:
        m = tr[col].notna()
        assert np.allclose(tr.loc[m, col], tr.loc[m, "I"])


def test_J_is_exactly_reconstructable():
    tr, _ = load_raw()
    spec = fit_cleaning_spec(tr)
    assert spec["j_max_residual"] < 1e-6


def test_cleaning_fixes_A_and_fills_J():
    tr, te = load_raw()
    spec = fit_cleaning_spec(tr)
    for d in (tr, te.drop(columns=[cfg.TARGET])):
        c = clean(d, spec)
        assert c["A"].abs().max() < cfg.A_SCALE_THRESHOLD
        assert c[cfg.CLEAN_FEATURES].notna().all().all()
        assert not {"D", "F", "K"} & set(c.columns)


def test_clean_rejects_unknown_genre():
    tr, _ = load_raw()
    spec = fit_cleaning_spec(tr)
    bad = tr.head(5).copy(); bad.loc[bad.index[0], cfg.TARGET] = "Documentary"
    try:
        clean(bad, spec)
        raise AssertionError("should have failed")
    except ValueError:
        pass


def test_split_is_stratified_and_disjoint(data):
    tr, ho = data["train"], data["holdout"]
    assert len(tr) == 480 and len(ho) == 120
    assert set(tr.show_id).isdisjoint(ho.show_id)
    p1 = tr[cfg.TARGET].value_counts(normalize=True); p2 = ho[cfg.TARGET].value_counts(normalize=True)
    assert (p1 - p2).abs().max() < .02

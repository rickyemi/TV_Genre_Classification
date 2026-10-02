"""
Stage 1: audit, clean and split the raw business-unit feeds.

The raw files are unmastered and unvalidated, so the audit runs first and every
cleaning rule below is backed by evidence it produces (see notebook 01):

1. Duplicate columns.  D, F and K are exact copies of I wherever present
   (correlation 1.000, identical values). D and F also have gaps that I fills.
   They are dropped, and I is kept as the single complete copy.
2. Unit error in A.  About 4-6% of A values are 100x too large: no value lies
   between |A| = 5.2 and 12.2, and dividing the large values by 100 gives the
   same distribution as the clean ones (KS p = 0.43). They are divided by 100
   and flagged in A_was_rescaled.
3. Missing J.  J is an exact linear combination of B, C, I, L, M and N
   (residual < 1e-8), so missing J values are reconstructed exactly. The
   coefficients are fitted on the TRAINING file only and saved to
   data/processed/cleaning_spec.json.
4. Labels are normalised (strip, title case) and checked against the three
   allowed genres. Exact duplicate shows are checked.

The labelled shows are then split 80 / 20 (stratified by genre) into train and
holdout. The 400 unlabelled shows are cleaned with the same spec.

Run:  python -m src.data.make_dataset
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src import config as cfg
from src.utils import get_logger, load_json, save_json

log = get_logger(__name__)


def load_raw():
    tr = pd.read_csv(cfg.RAW_TRAIN)
    te = pd.read_csv(cfg.RAW_TEST)
    for d in (tr, te):
        if d.columns.tolist() != cfg.RAW_FEATURES + [cfg.TARGET]:
            raise ValueError(f"Unexpected columns: {d.columns.tolist()}")
    log.info("Loaded raw training %s and test %s", tr.shape, te.shape)
    return tr, te


def audit(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Data-quality table: one row per finding with the evidence behind it."""
    rows = []
    for name, d in [("training", tr), ("test", te)]:
        X = d[cfg.RAW_FEATURES]
        for col, ref in cfg.DUPLICATE_OF.items():
            m = d[col].notna() & d[ref].notna()
            rows.append({"file": name, "check": f"{col} duplicates {ref}",
                         "value": f"{np.isclose(d.loc[m, col], d.loc[m, ref]).mean():.3f} identical",
                         "detail": f"{d[col].isna().sum()} missing in {col}, {d[ref].isna().sum()} in {ref}"})
        big = d["A"].abs() >= cfg.A_SCALE_THRESHOLD
        rows.append({"file": name, "check": "A x100 unit error", "value": f"{big.sum()} rows ({big.mean():.1%})",
                     "detail": f"no |A| between {d.loc[~big, 'A'].abs().max():.2f} and {d.loc[big, 'A'].abs().min():.2f}"})
        for col in cfg.RAW_FEATURES:
            if d[col].isna().any():
                rows.append({"file": name, "check": f"missing {col}", "value": f"{d[col].isna().sum()} rows",
                             "detail": f"{d[col].isna().mean():.1%}"})
        rows.append({"file": name, "check": "duplicate shows", "value": str(X.duplicated().sum()), "detail": ""})
        decimals = d["A"].astype(str).str.split(".").str[-1].str.len()
        rows.append({"file": name, "check": "stored precision (A)", "value": f"{decimals.median():.0f} decimals",
                     "detail": "test file was exported at lower precision" if name == "test" else ""})
    # Independent check of the x100 rule: A is ~99.7% a linear function of C, I, L, M, N,
    # so the raw value of a flagged row divided by its predicted value should be ~100.
    r = a_ratio_check(tr, te)
    rows.append({"file": "both", "check": "A x100 confirmed by regression", "value": f"median ratio {r['median']:.1f}",
                 "detail": f"IQR {r['q25']:.1f}-{r['q75']:.1f}; R2 of A on C,I,L,M,N = {r['r2']:.4f}"})
    labels = tr[cfg.TARGET].astype(str).str.strip().str.title()
    rows.append({"file": "training", "check": "genre labels", "value": str(labels.value_counts().to_dict()),
                 "detail": f"invalid: {sorted(set(labels) - set(cfg.CLASSES)) or 'none'}"})
    rows.append({"file": "test", "check": "genre labels", "value": f"{te[cfg.TARGET].isna().sum()} missing",
                 "detail": "unknown by design"})
    return pd.DataFrame(rows)


def a_ratio_check(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """raw A / A predicted from C, I, L, M, N for the rows flagged as x100 errors."""
    both = pd.concat([tr, te], ignore_index=True)
    src = ["C", "I", "L", "M", "N"]
    big = both["A"].abs() >= cfg.A_SCALE_THRESHOLD
    lr = LinearRegression().fit(both.loc[~big, src], both.loc[~big, "A"])
    ratio = both.loc[big, "A"] / lr.predict(both.loc[big, src])
    return {"median": float(ratio.median()), "q25": float(ratio.quantile(.25)), "q75": float(ratio.quantile(.75)),
            "r2": float(lr.score(both.loc[~big, src], both.loc[~big, "A"])), "ratios": ratio.values}


def fit_cleaning_spec(tr: pd.DataFrame) -> dict:
    """Learn the J reconstruction on the training file only."""
    d = tr.dropna(subset=["J"] + cfg.J_SOURCES)
    lr = LinearRegression().fit(d[cfg.J_SOURCES], d["J"])
    resid = np.abs(d["J"] - lr.predict(d[cfg.J_SOURCES])).max()
    if resid > 1e-6:
        raise ValueError(f"J is no longer an exact combination (max residual {resid:.2e})")
    return {"drop_duplicates_of": cfg.DUPLICATE_OF, "a_threshold": cfg.A_SCALE_THRESHOLD,
            "a_factor": cfg.A_SCALE_FACTOR, "j_sources": cfg.J_SOURCES,
            "j_coef": dict(zip(cfg.J_SOURCES, lr.coef_.tolist())), "j_intercept": float(lr.intercept_),
            "j_max_residual": float(resid)}


def clean(d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply the cleaning spec to raw rows (works on any new feed with the same columns)."""
    d = d.copy()
    for col, ref in spec["drop_duplicates_of"].items():           # fill the kept copy, then drop
        d[ref] = d[ref].fillna(d[col])
    d = d.drop(columns=list(spec["drop_duplicates_of"]))
    big = d["A"].abs() >= spec["a_threshold"]
    d["A_was_rescaled"] = big.astype(int)
    d.loc[big, "A"] = d.loc[big, "A"] / spec["a_factor"]
    miss = d["J"].isna() & d[spec["j_sources"]].notna().all(axis=1)
    d["J_was_reconstructed"] = miss.astype(int)
    d.loc[miss, "J"] = spec["j_intercept"] + sum(d.loc[miss, c] * w for c, w in spec["j_coef"].items())
    if cfg.TARGET in d and d[cfg.TARGET].notna().any():
        d[cfg.TARGET] = d[cfg.TARGET].astype(str).str.strip().str.title()
        bad = set(d[cfg.TARGET]) - set(cfg.CLASSES)
        if bad:
            raise ValueError(f"Unknown genres: {bad}")
    return d


def load(name: str) -> pd.DataFrame:
    return pd.read_csv({"train": cfg.TRAIN_FILE, "holdout": cfg.HOLDOUT_FILE, "unlabelled": cfg.UNLABELLED_FILE,
                        "labelled": cfg.CLEAN_LABELLED}[name])


def get_xy(df: pd.DataFrame, features):
    X = df[features].copy()
    y = df[cfg.TARGET].map({c: i for i, c in enumerate(cfg.CLASSES)}) if cfg.TARGET in df else None
    return X, y


def main():
    cfg.ensure_dirs()
    tr, te = load_raw()
    audit(tr, te).to_csv(cfg.TABLES_DIR / "data_quality_audit.csv", index=False)
    spec = fit_cleaning_spec(tr)
    save_json(spec, cfg.CLEANING_SPEC_FILE)
    lab, unl = clean(tr, spec), clean(te.drop(columns=[cfg.TARGET]), spec)
    lab.insert(0, "show_id", [f"TR{i:04d}" for i in range(len(lab))])
    unl.insert(0, "show_id", [f"TE{i:04d}" for i in range(len(unl))])
    lab.to_csv(cfg.CLEAN_LABELLED, index=False)
    unl.to_csv(cfg.CLEAN_UNLABELLED, index=False)
    train, hold = train_test_split(lab, test_size=cfg.HOLDOUT_SIZE, stratify=lab[cfg.TARGET],
                                   random_state=cfg.RANDOM_STATE)
    train.to_csv(cfg.TRAIN_FILE, index=False)
    hold.to_csv(cfg.HOLDOUT_FILE, index=False)
    unl.to_csv(cfg.UNLABELLED_FILE, index=False)
    log.info("Cleaned: A rescaled %d/%d labelled, %d/%d unlabelled | J reconstructed %d, %d | missing left %d",
             lab.A_was_rescaled.sum(), len(lab), unl.A_was_rescaled.sum(), len(unl),
             lab.J_was_reconstructed.sum(), unl.J_was_reconstructed.sum(),
             int(lab[cfg.CLEAN_FEATURES].isna().sum().sum() + unl[cfg.CLEAN_FEATURES].isna().sum().sum()))
    log.info("Split: train %d, holdout %d, unlabelled %d | train genres %s", len(train), len(hold), len(unl),
             train[cfg.TARGET].value_counts().to_dict())


if __name__ == "__main__":
    main()

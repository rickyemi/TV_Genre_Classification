"""
End-to-end pipeline runner (the same stages the Makefile calls).

    python -m src.pipeline                  # all stages (~6 minutes on 2 CPU cores)
    python -m src.pipeline --skip-stress
    python -m src.pipeline --stages data eda

Stages, in order:
    data        audit + clean raw feeds (duplicates, A unit error, J reconstruction), 80/20 split
    eda         exploratory analysis and data-quality evidence (figures 01-11)
    features    missing values, outliers, z-score, correlation / VIF, feature selection (12-16)
    train       optimisation steps + RandomizedSearchCV: SSGMM, XGBoost, SVM-RBF (+ screened DeepMLP) (17-18)
    evaluate    Train/Test comparison, per-genre metrics, ROC, calibration, importance (19-25)
    stress      11 stress tests + scorecard (26-27)
    predict     genre + probabilities for the 400 unlabelled shows (28)
    regression  J from C + two uncorrelated supporting features (29-32)
"""
import argparse
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

from src.utils import get_logger  # noqa: E402

log = get_logger("pipeline")
STAGES = ["data", "eda", "features", "train", "evaluate", "stress", "predict", "regression"]


def run_stage(name: str):
    if name == "data":
        from src.data.make_dataset import main; main()
    elif name == "eda":
        from src.analysis.eda import run; run()
    elif name == "features":
        from src.features.build_features import run; run()
    elif name == "train":
        from src.models.train_model import run; run()
    elif name == "evaluate":
        from src.models.evaluate_model import run; run()
    elif name == "stress":
        from src.models.stress_test import run; run()
    elif name == "predict":
        from src.models.predict_model import main; main()
    elif name == "regression":
        from src.models.regression_j import run; run()


def main():
    p = argparse.ArgumentParser(description="TV genre classification pipeline")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--skip-stress", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    for s in [s for s in a.stages if not (a.skip_stress and s == "stress")]:
        t = time.time(); log.info("==== stage: %s ====", s); run_stage(s)
        log.info("stage %s finished in %.0fs", s, time.time() - t)
    log.info("Pipeline complete in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()

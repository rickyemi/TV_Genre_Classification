"""
Stage 7: classify the shows of unknown genre with probability scores.

Uses the PRODUCTION models (same tuned hyperparameters, refit on all 600
labelled shows). For every show the output gives:

    P(Comedy), P(Drama), P(Reality) for each of the three models
    Predicted_Genre      from the recommended model: the classifier whose weakest
                         target metric is highest on the repeated-CV Test
                         estimate (read from reports/tables; SSGMM if missing)
    Confidence           its highest class probability
    Model_Agreement      how many of the 3 models agree with Predicted_Genre
    Needs_Review         True when confidence < 0.60 or the models disagree

Usage
-----
    python -m src.models.predict_model                                  # the 400 test shows
    python -m src.models.predict_model --input raw_new_feed.csv --output out.csv

New raw feeds go through the same cleaning spec first (duplicate columns,
A unit error, J reconstruction), so they can be passed exactly as received.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.data.make_dataset import clean
from src.models.train_model import load_models
from src.utils import get_logger, load_json
from src.visualization.visualize import save_fig

log = get_logger(__name__)
DEFAULT_RECOMMENDED = "SSGMM"


def recommended_model() -> str:
    f = cfg.TABLES_DIR / "model_comparison_train_test.csv"
    if not f.exists():
        return DEFAULT_RECOMMENDED
    t = pd.read_csv(f)
    t = t[(t.Set == "CV Test") & t.Model.isin(cfg.MODEL_NAMES)].set_index("Model")
    return str(t[cfg.TARGET_METRICS].min(axis=1).idxmax()) if len(t) else DEFAULT_RECOMMENDED


RECOMMENDED = recommended_model()


def predict(df: pd.DataFrame) -> pd.DataFrame:
    feats = load_json(cfg.MODEL_METADATA_FILE)["features"]
    models = load_models("production")
    out = pd.DataFrame({"show_id": df["show_id"].values if "show_id" in df else
                        [f"S{i:04d}" for i in range(len(df))]})
    calls = {}
    for name, m in models.items():
        P = m.predict_proba(df[feats])
        key = name.replace("-", "_")
        for k, c in enumerate(cfg.CLASSES):
            out[f"{key}_P_{c}"] = P[:, k].round(4)
        calls[name] = np.array(cfg.CLASSES)[P.argmax(1)]
        out[f"{key}_Genre"] = calls[name]
    rec = RECOMMENDED.replace("-", "_")
    out["Predicted_Genre"] = calls[RECOMMENDED]
    out["Confidence"] = out[[f"{rec}_P_{c}" for c in cfg.CLASSES]].max(axis=1).round(4)
    agree = sum((calls[n] == calls[RECOMMENDED]).astype(int) for n in cfg.MODEL_NAMES)
    n = len(cfg.MODEL_NAMES)
    out["Model_Agreement"] = [f"{a}/{n}" for a in agree]
    out["Needs_Review"] = (out.Confidence < cfg.REVIEW_CONFIDENCE) | (agree < n)
    return out


def plot_predictions(res):
    rec = RECOMMENDED.replace("-", "_")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    c = res.Predicted_Genre.value_counts().reindex(cfg.CLASSES, fill_value=0)
    ax.bar(cfg.CLASSES, c.values, color=[cfg.COLORS[g] for g in cfg.CLASSES], width=.6)
    for i, v in enumerate(c.values):
        ax.text(i, v + 4, f"{v} ({v / len(res):.0%})", ha="center", fontsize=9)
    ax.set_ylim(0, c.max() * 1.18); ax.set_ylabel("Shows"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Predicted genres for {len(res)} shows ({RECOMMENDED})", fontsize=10)
    ax = axes[1]
    for g in cfg.CLASSES:
        ax.hist(res.loc[res.Predicted_Genre == g, "Confidence"], bins=np.linspace(1 / 3, 1, 15), histtype="step",
                lw=2, color=cfg.COLORS[g], label=g)
    ax.axvline(cfg.REVIEW_CONFIDENCE, color="#e34948", ls="--", lw=1)
    ax.set_xlabel("Confidence (max class probability)"); ax.set_ylabel("Shows"); ax.legend()
    ax.set_title("Confidence by predicted genre (red = review threshold)", fontsize=10)
    ax = axes[2]
    tern = res[[f"{rec}_P_{g}" for g in cfg.CLASSES]].values
    xy = np.c_[tern[:, 1] + tern[:, 2] / 2, tern[:, 2] * np.sqrt(3) / 2]
    ax.plot([0, 1, .5, 0], [0, 0, np.sqrt(3) / 2, 0], color=cfg.NEUTRAL, lw=1)
    for g in cfg.CLASSES:
        m = (res.Predicted_Genre == g).values
        ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=.65, color=cfg.COLORS[g], label=g, edgecolor="none")
    for (x, y), g in zip([(-.04, -.05), (1.04, -.05), (.5, .9)], cfg.CLASSES):
        ax.text(x, y, g, ha="center", fontsize=8.5)
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Probability simplex (corners = certain)", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "28_test_set_predictions",
                    f"{int(res.Needs_Review.sum())} of {len(res)} shows flagged for manual review (confidence < 0.60 or models disagree)")


def main(input_path=None, output_path=cfg.PREDICTIONS_FILE):
    cfg.ensure_dirs()
    if input_path is None:
        df = pd.read_csv(cfg.UNLABELLED_FILE)          # already cleaned by the data stage
    else:
        raw = pd.read_csv(input_path)
        df = clean(raw.drop(columns=[cfg.TARGET], errors="ignore"), load_json(cfg.CLEANING_SPEC_FILE))
    res = predict(df)
    res.to_csv(output_path, index=False)
    if input_path is None:
        plot_predictions(res)
    log.info("Classified %d shows -> %s | review %d | agreement %s", len(res), res.Predicted_Genre.value_counts().to_dict(),
             int(res.Needs_Review.sum()), res.Model_Agreement.value_counts().to_dict())
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Classify TV shows into Comedy / Drama / Reality")
    p.add_argument("--input", default=None, help="raw CSV with columns A-O (default: the 400 test shows)")
    p.add_argument("--output", default=str(cfg.PREDICTIONS_FILE))
    a = p.parse_args()
    main(a.input, Path(a.output))

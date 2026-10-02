"""
Central configuration for the TV-genre classification project.

Paths, column roles, data-cleaning rules, tuning budgets, acceptance criteria
and plot colours live here so scripts, notebooks and tests all agree.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (built from this file, so the project runs from any working directory)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR, INTERIM_DIR = DATA_DIR / "raw", DATA_DIR / "interim"
PROCESSED_DIR, EXTERNAL_DIR = DATA_DIR / "processed", DATA_DIR / "external"
MODELS_DIR, REPORTS_DIR = ROOT_DIR / "models", ROOT_DIR / "reports"
FIGURES_DIR, TABLES_DIR = REPORTS_DIR / "figures", REPORTS_DIR / "tables"

RAW_TRAIN = RAW_DIR / "tv_media_training_data.csv"      # 600 labelled shows
RAW_TEST = RAW_DIR / "tv_media_test_data.csv"           # 400 shows, genre unknown
CLEAN_LABELLED = INTERIM_DIR / "labelled_clean.csv"
CLEAN_UNLABELLED = INTERIM_DIR / "unlabelled_clean.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"                # 80% of the labelled shows
HOLDOUT_FILE = PROCESSED_DIR / "holdout.csv"            # 20% held out for Test metrics
UNLABELLED_FILE = PROCESSED_DIR / "unlabelled.csv"      # the 400 shows to classify
CLEANING_SPEC_FILE = PROCESSED_DIR / "cleaning_spec.json"
SELECTED_FEATURES_FILE = PROCESSED_DIR / "selected_features.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
PREDICTIONS_FILE = REPORTS_DIR / "test_set_genre_predictions.csv"

# The three classifiers (all built on the 5-D latent basis and the semi-supervised Gaussian mixture)
MODEL_NAMES = ["SSGMM", "XGBoost", "SVM-RBF"]
MODEL_LABELS = {"SSGMM": "Semi-supervised Gaussian mixture",
                "XGBoost": "XGBoost on mixture posteriors + inputs",
                "SVM-RBF": "RBF-SVM on mixture posteriors"}
SCREENED_MODELS = ["DeepMLP"]     # also optimised and screened; ranked 4th, so not deployed (see train stage)


def model_file(name: str, stage: str = "production") -> Path:
    """stage = 'evaluation' (fit on the 80% split) or 'production' (refit on all 600)."""
    return MODELS_DIR / f"{stage}_{name.lower().replace('-', '_')}.joblib"


# --------------------------------------------------------------------------- #
# Columns and classes
# --------------------------------------------------------------------------- #
TARGET = "genre"
CLASSES = ["Comedy", "Drama", "Reality"]
RAW_FEATURES = list("ABCDEFGHIJKLMNO")
DUPLICATE_OF = {"D": "I", "F": "I", "K": "I"}   # exact copies found in the audit (I is complete)
A_SCALE_THRESHOLD = 10.0                         # |A| >= 10 is a x100 unit error (gap 5.2 -> 12.2)
A_SCALE_FACTOR = 100.0
J_SOURCES = ["B", "C", "I", "L", "M", "N"]       # J is an exact linear combination of these

# Features after cleaning (duplicates removed, A rescaled, J reconstructed) + flag
CLEAN_FEATURES = ["A", "B", "C", "E", "G", "H", "I", "J", "L", "M", "N", "O"]
FLAG_FEATURES = ["A_was_rescaled"]

# --------------------------------------------------------------------------- #
# Splits, tuning
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
CV_FOLDS = 5
N_ITER = {"SSGMM": 24, "XGBoost": 40, "SVM-RBF": 40, "DeepMLP": 12}
MAX_TRAIN_VAL_GAP = 0.05          # overfitting guard on the CV tuning score
SCORING = "min_weighted"          # tuning score = the WORST of the five weighted target metrics
CV_REPEATS = 5                    # repeated 5-fold CV on all 600 shows for the CV Train/Test estimate

# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
FDR_ALPHA = 0.05
MIN_MUTUAL_INFO = 0.01
VIF_MAX = 1000.0                  # VIF above this = exact linear redundancy
LATENT_BASIS = ["C", "I", "L", "M", "N"]   # 5 independent columns spanning B, C, I, J, L, M, N (rank 5)

# --------------------------------------------------------------------------- #
# Acceptance criteria (fixed before any model was trained)
# --------------------------------------------------------------------------- #
# Target set by the business: Accuracy, Sensitivity, Specificity, Precision and F1 >= 0.89 on BOTH Train and
# Test. Sensitivity, Specificity, Precision and F1 are one-vs-rest per genre, averaged with weights equal to
# each genre's share of shows (support-weighted). Macro (unweighted) averages and per-genre values are reported
# alongside as secondary endpoints.
TARGET_METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
TARGET_THRESHOLD = 0.89
ACCEPTANCE = {
    "metric_floor": TARGET_THRESHOLD,   # every TARGET_METRIC (weighted) on Train and on Test
    "max_train_test_gap": 0.05,         # accuracy gap, overfitting check
}
REVIEW_CONFIDENCE = 0.60          # predictions below this max-probability go to manual review

# --------------------------------------------------------------------------- #
# Plot style (validated colour-blind-safe palette, fixed order)
# --------------------------------------------------------------------------- #
COLORS = {"Comedy": "#2a78d6", "Drama": "#eb6834", "Reality": "#1baf7a"}
MODEL_COLORS = {"SSGMM": "#008300", "XGBoost": "#4a3aa7", "SVM-RBF": "#e87ba4", "DeepMLP": "#c98500"}
NEUTRAL, TEXT_PRIMARY, TEXT_SECONDARY = "#8a8985", "#0b0b0b", "#52514e"
GRID, SURFACE = "#e4e3df", "#fcfcfb"
SEQUENTIAL_CMAP = ["#f2f7fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
DIVERGING_CMAP = ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b52f2f"]


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

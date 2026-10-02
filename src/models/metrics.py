"""
Classification metrics used everywhere in the project (tuning, evaluation,
stress tests, tests).

For three genres every metric is computed one-vs-rest per genre:
    Sensitivity (recall)   TP / (TP + FN)
    Specificity            TN / (TN + FP)
    Precision (PPV)        TP / (TP + FP)
    NPV                    TN / (TN + FN)
    F1                     2 TP / (2 TP + FP + FN)
and then averaged two ways:
    weighted   weights = each genre's share of shows (the business target)
    macro      plain mean over genres (every genre counts equally)
Accuracy is the share of shows assigned the right genre (identical to
weighted sensitivity).

`min_weighted_score` (the tuning score) is the lowest of the five weighted
target metrics, so RandomizedSearchCV optimises the metric that is furthest
from the >= 0.89 target instead of only one of them.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

from src import config as cfg

K = len(cfg.CLASSES)
OVR = ["Sensitivity", "Specificity", "Precision", "NPV", "F1"]


def per_class_from_confusion(cm: np.ndarray) -> dict:
    n = cm.sum(); out = {}
    for k, c in enumerate(cfg.CLASSES):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp; tn = n - tp - fn - fp
        sd = lambda a, b: a / b if b else 0.0  # noqa: E731
        out[c] = {"Sensitivity": sd(tp, tp + fn), "Specificity": sd(tn, tn + fp), "Precision": sd(tp, tp + fp),
                  "NPV": sd(tn, tn + fn), "F1": sd(2 * tp, 2 * tp + fp + fn), "n": int(cm[k].sum())}
    return out


def summary_metrics(y, pred) -> dict:
    """Accuracy plus weighted and macro one-vs-rest metrics."""
    cm = confusion_matrix(np.asarray(y), np.asarray(pred), labels=range(K))
    per = per_class_from_confusion(cm)
    w = cm.sum(1) / cm.sum()
    out = {"Accuracy": float(np.trace(cm) / cm.sum())}
    for m in OVR:
        v = np.array([per[c][m] for c in cfg.CLASSES])
        out[m] = float(w @ v)                 # weighted (target)
        out[f"{m}_macro"] = float(v.mean())   # macro (secondary)
    out["min_target"] = min(out[m] for m in cfg.TARGET_METRICS)
    out["per_class"], out["confusion"] = per, cm
    return out


def min_weighted_score(y, pred) -> float:
    return summary_metrics(y, pred)["min_target"]


MIN_WEIGHTED_SCORER = make_scorer(min_weighted_score)

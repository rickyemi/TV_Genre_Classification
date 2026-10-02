# Model card: TV genre classifiers and the J regression

| Item | Detail |
|---|---|
| Classifiers | `production_ssgmm.joblib` (recommended), `production_xgboost.joblib`, `production_svm_rbf.joblib`: refit on all 600 labelled shows |
| Evaluation copies | `evaluation_*.joblib`: fit on the 480-show training split (holdout Train/Test metrics); `evaluation_deepmlp.joblib` is the screened, non-deployed MLP |
| Model types | SSGMM = semi-supervised class-conditional Gaussian mixture (2 full-covariance components per genre, EM over labelled + 400 unlabelled shows); XGBoost and SVM-RBF are trained on the mixture's clipped log-posteriors (XGBoost also on the inputs). Code: `src/models/gmm_models.py` |
| Regression | `regression_j.joblib` (cubic polynomial on C, B', M'; the pipeline orthogonalises B and M internally, so pass raw C, B, M) and `regression_j_ols.joblib` |
| Metadata | `model_metadata.json` (features, tuned hyperparameters, CV scores, gap-guard flag), `regression_j_metadata.json` |
| Input | Cleaned features C, I, L, M, N. Raw feeds go through `src.data.make_dataset.clean` first; `predict_model --input` does this automatically. Blank inputs are allowed (marginalised) |
| Output | P(Comedy), P(Drama), P(Reality) per model, predicted genre, confidence, model agreement, review flag |
| Versions | Python 3.11, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.14.0 (pinned in `requirements.txt`) |

## Loading
```python
import joblib, pandas as pd
from src.data.make_dataset import clean
from src.utils import load_json
spec = load_json("data/processed/cleaning_spec.json")      # created by `make data`
feats = load_json("models/model_metadata.json")["features"]   # ['C', 'I', 'L', 'M', 'N']
model = joblib.load("models/production_ssgmm.joblib")       # needs the project on sys.path (`pip install -e .`)
shows = clean(pd.read_csv("raw_feed.csv").drop(columns=["genre"], errors="ignore"), spec)
proba = model.predict_proba(shows[feats])                     # columns: Comedy, Drama, Reality

reg = joblib.load("models/regression_j.joblib")
j_hat = reg["model"].predict(shows[reg["features"]])          # features: C, B, M
```

## Performance and fitness for use
* Target: Accuracy, Sensitivity, Specificity, Precision, F1 (support-weighted one-vs-rest) ≥ 0.89 on Train and Test.
* **Met in repeated 5×5 CV on all 600 shows** by all three models (Test 0.898–0.907, Train 0.903–0.921, gaps ≤ 0.02).
* **Not met on the fixed 120-show holdout** (Test 0.85–0.88); bootstrap 95% CIs include 0.89, and about 6 of the 14 to 17 holdout errors look like mislabels.
* Macro (unweighted) metrics are lower (CV macro F1 0.86–0.87); Reality recall is 0.72–0.76. Use the review flag for Reality-targeted campaigns.
* SSGMM probabilities are sharp (holdout log-loss 1.16 vs 0.44–0.45 for XGBoost and SVM-RBF): use its genre call, but prefer the SVM-RBF or XGBoost probabilities when calibrated scores matter.
* J regression: Test RMSE 0.263, MAE 0.203, adjusted R² 0.964 (C only: 1.398 / 1.078 / −0.010).

## Known limitations
* **Measurement precision:** the high accuracy relies on thin cluster shapes in the 5-D basis. Gaussian noise of 0.1 SD costs 0.12–0.17 accuracy. Feeds must keep their current precision.
* **Missing inputs:** +20% blank cells costs 0.10–0.18 accuracy even with marginalisation; enforce completeness checks upstream.
* 600 labelled shows, only 106 Reality; 36 training labels disagree with confident out-of-fold predictions and should be re-checked.
* The cleaning rules (A ×100, J identity, duplicate columns) and the latent basis are derived from this data. Re-run `make data features` on every new feed.
* B' and M' in the regression are residuals defined by the Train-split fit; their correlation with C is 0 on Train and up to 0.12 on Test.

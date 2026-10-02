# TV Show Genre Classification (Comedy / Drama / Reality) from Unvalidated Business-Unit Features

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Models](https://img.shields.io/badge/models-SSGMM%20%7C%20XGBoost%20%7C%20SVM--RBF-orange)

**Author:** YO ([@rickyemi](https://github.com/rickyemi))

## Goal
Several business units supply anonymised features (A–O) for each TV show. Marketing campaigns depend on classifying shows as **Comedy, Drama or Reality**, so accuracy is critical. The project (1) audits and cleans the raw, unvalidated inputs, (2) optimises three classifiers to reach **Accuracy, Sensitivity, Specificity, Precision and F1 ≥ 0.89 on Train and Test**, (3) classifies the 400 shows of unknown genre with probability scores, and (4) predicts **Feature J from Feature C plus two uncorrelated supporting features**.

## Data
| Item | Value |
|---|---|
| Files | `tv_media_training_data.csv`: 600 shows with genre · `tv_media_test_data.csv`: 400 shows, genre unknown |
| Genres (training) | Comedy 367 (61%) · Drama 127 (21%) · Reality 106 (18%), so always guessing Comedy scores 61% |
| Data-quality findings | **D, F, K are exact copies of I** · **~5% of A values carry a ×100 unit error** (gap 5.2–12.2; confirmed by KS test and a regression ratio ≈ 100) · **J is missing in 7%** but is an **exact linear combination** of B, C, I, L, M, N · B…N span only **5 dimensions** · E, H, O carry no genre signal · no drift between files (adversarial AUC 0.55) |
| Split | Labelled shows 80/20 stratified: Train 480 / Test (holdout) 120; plus repeated 5×5 CV on all 600 |

## Methodology
| Stage | What was done |
|---|---|
| **Data preprocessing** | Cleaning spec fitted on the training file: drop D/F/K (keep complete I) · divide A by 100 where abs(A) ≥ 10 (flagged) · rebuild missing J exactly, leaving 0 missing values. Blank inputs in future feeds are **marginalised** by the Gaussian mixture (no imputation) |
| **Feature selection** | FDR, mutual-information and VIF rules plus a rank check; five strategies compared by repeated CV → **5-D latent basis C, I, L, M, N** (reproduces B and J with R² = 1) |
| **Optimisation steps** | (1) latent basis · (2) class-conditional Gaussian mixture, 2 full-covariance components per genre · (3) **semi-supervised EM** with the 400 unlabelled shows (feature values only) · (4) XGBoost and RBF-SVM stacked on the mixture's clipped log-posteriors · (5) RandomizedSearchCV, 5-fold, score = **worst of the 5 weighted target metrics**, gap guard ≤ 0.05 |
| **Classifiers** | **SSGMM** (24 candidates) · **XGBoost** on posteriors + inputs (40) · **SVM-RBF** on posteriors (40, Platt probabilities). A deep MLP was also optimised (12) and screened out (4th). Evaluation copies fit on 480 shows; production copies refit on 600; joblib |
| **Performance evaluation** | Accuracy and one-vs-rest Sensitivity, Specificity, Precision, F1 (support-weighted = target; macro and per-genre = secondary), plus AUC, log-loss; Train vs Test for the holdout and for repeated CV; 11 stress tests |
| **Acceptance criteria** | Every target metric ≥ 0.89 on Train and Test; Train − Test accuracy gap ≤ 0.05 |
| **Primary endpoints** | Weighted Accuracy, Sensitivity, Specificity, Precision, F1 on Train and Test |
| **Secondary endpoints** | Macro and per-genre metrics, AUC, log-loss; bootstrap CI; stress tests; J regression MSE, RMSE, MAE, adjusted R² (Train vs Test) |

## Results
**Train vs Test, repeated 5×5 CV on all 600 shows** (support-weighted one-vs-rest; bold = meets ≥ 0.89)

| Metric | SSGMM Train | SSGMM Test | XGBoost Train | XGBoost Test | SVM-RBF Train | SVM-RBF Test |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | **0.907** | **0.904** | **0.921** | **0.901** | **0.911** | **0.903** |
| Sensitivity | **0.907** | **0.904** | **0.921** | **0.901** | **0.911** | **0.903** |
| Specificity | **0.908** | **0.907** | **0.910** | **0.900** | **0.903** | **0.902** |
| Precision | **0.906** | **0.902** | **0.921** | **0.899** | **0.911** | **0.902** |
| F1 | **0.905** | **0.902** | **0.919** | **0.898** | **0.908** | **0.900** |
| F1 (macro, secondary) | 0.874 | 0.869 | 0.893 | 0.862 | 0.878 | 0.864 |

**Train vs Test, fixed holdout** (Train 480 / Test 120)

| Metric | SSGMM Train | SSGMM Test | XGBoost Train | XGBoost Test | SVM-RBF Train | SVM-RBF Test |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | **0.917** | 0.858 | **0.933** | 0.875 | **0.917** | 0.883 |
| Sensitivity | **0.917** | 0.858 | **0.933** | 0.875 | **0.917** | 0.883 |
| Specificity | **0.919** | 0.867 | **0.928** | 0.861 | **0.913** | 0.862 |
| Precision | **0.915** | 0.852 | **0.933** | 0.872 | **0.916** | 0.883 |
| F1 | **0.915** | 0.852 | **0.932** | 0.868 | **0.915** | 0.875 |

* **Target met in repeated CV:** all 33 checks pass (3 models × 5 metrics × Train/Test + gap). The optimisation lifted the weakest metric from **0.78** (previous RBF-SVM) to **0.90** (figure 18); the Gaussian-mixture step alone added about 0.10.
* **Target not met on the fixed 120-show holdout** (Test 0.85–0.88; 16 of 33 checks). This holdout is harder than average: a mixture fitted with these 120 shows included still misclassifies 14 of them, and 6 of those have P(label) ≈ 0 (likely mislabels). The bootstrap 95% CI for SVM-RBF holdout accuracy is 0.83–0.94, which contains 0.89.
* **Weighted, not macro:** macro F1 is about 0.86–0.87 in CV. Comedy is recalled 0.97–0.98, Drama 0.82–0.83 and Reality 0.72–0.76.
* **Stress tests:** each model passes 7 of 9 rules (repeated CV SD 0.02, no drift, 10% label noise costs ≤ 0.02, edge cases pass). **Weak points:** sensitivity to measurement noise (0.1 SD of noise costs 0.12–0.17 accuracy) and to blanks (+20% blank cells costs 0.10–0.18, down from 0.32 with median imputation). 36 training labels are flagged for re-review.

**Predictions for the 400 unknown shows** (`reports/test_set_genre_predictions.csv`: P(Comedy), P(Drama), P(Reality) for every model; recommended model **SSGMM**, the highest weakest metric in CV)

| Predicted genre | Comedy | Drama | Reality | All 3 models agree | Needs review (confidence < 0.60 or disagreement) |
|---|---:|---:|---:|---:|---:|
| Shows | 271 (68%) | 78 (20%) | 51 (13%) | 389 (97%) | 32 (8%) |

**J regression** (930 shows with observed J from both files; Train 744 / Test 186). Supporting features **B' and M'**: B and M orthogonalised against C (and M against B'), so C, B', M' are mutually uncorrelated (r = 0 on Train, ≤ 0.12 on Test; VIF 11.2 → 1.0)

| Model | Features | Train MSE | Train RMSE | Train MAE | Train adj. R² | Test MSE | Test RMSE | Test MAE | Test adj. R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OLS baseline | C | 1.831 | 1.353 | 1.072 | 0.118 | 1.955 | 1.398 | 1.078 | −0.010 |
| OLS, uncorrelated as measured | C + G + M | 1.812 | 1.346 | 1.067 | 0.125 | 1.952 | 1.397 | 1.074 | −0.020 |
| OLS | C + B' + M' | 0.084 | 0.289 | 0.228 | 0.960 | 0.075 | 0.274 | 0.218 | 0.961 |
| **Polynomial (degree 3, best)** | C + B' + M' | **0.066** | **0.257** | **0.204** | **0.968** | **0.069** | **0.263** | **0.203** | **0.964** |
| Cubic spline + ridge | C + B' + M' | 0.073 | 0.271 | 0.215 | 0.965 | 0.078 | 0.279 | 0.217 | 0.959 |
| SVR-RBF | C + B' + M' | 0.075 | 0.274 | 0.224 | 0.964 | 0.079 | 0.281 | 0.225 | 0.959 |
| XGBoost | C + B' + M' | 0.061 | 0.248 | 0.186 | 0.970 | 0.102 | 0.320 | 0.239 | 0.947 |

* Features that are uncorrelated with C as measured (E, G, H, M, O) barely help (best pair G + M: Test RMSE 1.397). The information about J that C lacks sits in features correlated with C, so B and M were orthogonalised: same information, no multicollinearity. The cubic polynomial has the lowest CV RMSE (0.269) and cuts Test RMSE from 1.40 to **0.26** (adjusted R² −0.01 → **0.96**). J is still exactly 0.446·B + 0.337·C + 0.643·I + 0.360·L − 0.282·M + 0.971·N when all six features are available.

> **Recommendations:** deploy SSGMM with the review flag; re-check the 36 suspect labels and collect more Reality shows to lift the macro metrics; enforce precision and completeness checks on the feeds, since the 0.89 accuracy depends on precise inputs.

---

### Run it
```bash
make install      # pinned dependencies (CPU only)
make all          # data → eda → features → train → evaluate → stress → predict → regression (~6 min)
make test         # 14 unit / integration tests
make notebooks    # execute the 6 notebooks in place
python -m src.models.predict_model --input new_raw_feed.csv --output predictions.csv
docker build -t tv-genre-classification . && docker run --rm -v "$PWD":/app tv-genre-classification make all
```

### Repository layout
```
├── LICENSE · Makefile · README.md · requirements.txt · pyproject.toml
├── Dockerfile · docker-compose.yml · .github/workflows/ci.yml
├── data/        raw/ (2 original CSVs) · interim/ · processed/ · external/
├── docs/        TV_Genre_Classification_OnePager.docx
├── models/      production_*.joblib · evaluation_*.joblib · regression_j*.joblib · *.json · MODEL_CARD.md
├── notebooks/   01 EDA & data quality · 02 preprocessing & feature selection · 03 optimisation & training
│                04 evaluation & test predictions · 05 stress tests · 06 regression of J
├── references/  data_dictionary.md · methods_references.md
├── reports/     figures/ (32 PNGs) · tables/ · test_set_genre_predictions.csv
├── src/         config.py · pipeline.py · utils.py
│   ├── data/          make_dataset.py (audit + cleaning + split)
│   ├── analysis/      eda.py · stats.py
│   ├── features/      build_features.py
│   ├── models/        gmm_models.py · metrics.py · train_model.py · mlp_model.py · evaluate_model.py
│   │                  stress_test.py · predict_model.py · regression_j.py
│   └── visualization/ visualize.py
└── tests/       test_data.py · test_models.py
```

# Data dictionary and data-quality findings

**Files:** `data/raw/tv_media_training_data.csv` (600 shows, genre known) and `data/raw/tv_media_test_data.csv` (400 shows, genre unknown). One row is one show. The data are anonymised: feature meanings are not disclosed, so features are described by their observed behaviour.

| Column | Type | Raw issues found | Cleaning action | Used in model |
|---|---|---|---|---|
| A | numeric | 49 values (4.3% train, 5.8% test) are 100× too large | divided by 100; flagged in `A_was_rescaled` | no (99.6% predictable from the basis) |
| B | numeric | exact linear combination of C, I, J, L, M, N | none | classifier: no; J regression: yes, as B' (orthogonalised) |
| C | numeric | none | none | yes (latent basis); also the main regression predictor |
| D | numeric | exact copy of I; 9.3% / 6.0% missing | dropped | no |
| E | numeric | none (no genre signal) | none | no |
| F | numeric | exact copy of I; same gaps as D | dropped | no |
| G | numeric | none (weak signal) | none | no |
| H | numeric | none (no genre signal) | none | no |
| I | numeric | complete copy of D/F/K | kept as the single copy | yes (latent basis; weak alone, needed jointly) |
| J | numeric | 7.0% missing in both files; exact combination of B, C, I, L, M, N | missing values reconstructed exactly | no (redundant given the basis); regression target |
| K | numeric | exact copy of I | dropped | no |
| L, M, N | numeric | none | none | yes (latent basis); M also in the J regression, as M' |
| O | numeric | none (no genre signal) | none | no |
| genre | text | training: Comedy 367 (61%), Drama 127 (21%), Reality 106 (18%); test: all missing | trimmed, title case, validated | target |

## Evidence for the cleaning rules
* **Duplicates:** D, F and K equal I in 100% of rows where both are present (correlation 1.000). I has no gaps, so it is kept.
* **A unit error:** no absolute value of A lies between 5.2 and 12.2. The large values divided by 100 match the distribution of the normal ones (KS p = 0.43). A is 99.8% predictable from C, I, L, M and N, and for the flagged shows raw A / predicted A has a median of 99.9.
* **J identity:** J = 0.446·B + 0.337·C + 0.643·I + 0.360·L − 0.282·M + 0.971·N (fitted on the training file, maximum residual < 1e-14), so the 70 missing values are recovered exactly.
* **Rank:** B, C, I, J, L, M and N span only 5 independent dimensions. C, I, L, M, N form a basis: they reproduce B and J with R² = 1, so the classifiers use exactly these five.
* **Export precision:** the test file is stored with 9 decimals, the training file with ~16, so it went through a different export.
* **Drift:** no feature differs significantly between the labelled and unlabelled shows (KS, FDR q > 0.18); adversarial validation AUC ≈ 0.53.

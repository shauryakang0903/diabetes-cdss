# Development-only operating-threshold analysis

## Fixed data, features, and models

Only `data/processed/nhanes/nhanes_development_2011_2020.csv` is loaded:
25,580 adults, 3,690 physician-diagnosed diabetes positives from DIQ010,
and 21,890 negatives. The shared loader checks frozen counts, development cycle
labels, unique IDs, adult ages, and binary outcomes. Actual 2021-2023 temporal
data is never opened, inspected, or hashed. Development bytes are hashed before
and after execution; previous experiment results are preserved.

Core-10 is unchanged: age, sex, race_ethnicity, bmi, waist_cm,
mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl, total_cholesterol_mg_dl,
smoking_status. Numeric median imputation, categorical most-frequent imputation,
and OneHotEncoder(handle_unknown="ignore") remain inside each pipeline.
StandardScaler is used for LR numeric features only.

LR: L2 (l1_ratio=0 in installed sklearn 1.9), lbfgs, C=0.1,
max_iter=2000, class_weight=None, seed 42. XGBoost: 200 trees,
learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8,
colsample_bytree=0.8, reg_alpha=1, reg_lambda=10, binary:logistic objective,
gamma=0, scale_pos_weight=1, hist trees on CPU, eval_metric=logloss,
n_jobs=1, seed 42, no early stopping. Full estimator parameters and package
versions are recorded in provenance. The missing-value NaN model-parameter
sentinel is encoded as the string "NaN" for strict JSON. No new packages were
needed. The raw pipeline factory is reused from the calibration comparison;
no calibration wrapper is used or modified. There is no class/sample weighting,
new hyperparameter search, or full-development final fit.

## Leakage-safe OOF probabilities

StratifiedKFold: five folds, shuffle=True, random_state=42. Splits are generated
once and shared across models. A fresh full preprocessing/classifier pipeline
fits on 20,464 training participants and predicts only its 5,116 held-out
participants per fold. Every participant has exactly one OOF probability per
model. No preprocessing fit occurs before splitting, and no training or
full-development in-sample predictions enter this analysis.

The saved OOF file has 51,160 rows, uniquely identified by (model, seqn).
Columns also record nhanes_cycle, diabetes, outer_fold, and probability.

## Curves and metric conventions

There are 981 display thresholds per model: 0.010 through 0.990 inclusive,
step 0.001. Prediction rule: p >= threshold; tied scores are positive.
Sorted probabilities and cumulative outcomes give exact TP/FP/TN/FN counts.
Both outcome classes must be present.

- Sensitivity = TP/(TP+FN); specificity = TN/(TN+FP).
- Precision/PPV = TP/(TP+FP); NPV = TN/(TN+FN).
- Accuracy = (TP+TN)/N; F1 = 2TP/(2TP+FP+FN).
- Balanced accuracy = (sensitivity+specificity)/2.
- FPR = 1-specificity; FNR = 1-sensitivity.
- Predicted-positive fraction = (TP+FP)/N.
- Youden J = sensitivity+specificity-1.

Undefined PPV (no predicted positives) and NPV (no predicted negatives) are
saved as missing, not zero. Metrics are pooled OOF counts/rates, not averages
of fold metrics; nonlinear metrics such as PPV/F1 can therefore differ slightly
from previous mean-fold reports.

## Candidate operating points and deterministic rules

Selection examines **all unique OOF probabilities plus 0, 0.5, and 1**, rather
than restricting candidates to the display grid. Five references are reported:

1. Default 0.50, kept exactly fixed.
2. Minimum absolute distance from sensitivity 0.80.
3. Minimum absolute distance from sensitivity 0.90.
4. Maximum Youden J (a statistical reference).
5. Minimum absolute sensitivity-specificity difference.

Primary-criterion ties within 1e-12 favor higher specificity, then larger
threshold. The nearest sensitivity can be above or below target; this is not
an at-least-target constraint. Discrete/tied predictions may make exact targets
unattainable, so target, selection error, and achieved sensitivity are recorded.
Both models attained exactly 80% and 90% sensitivity in this run. No thresholds
outside [0,1] are used.

Reported thresholds are rounded for readability. Use full-precision candidate
CSV values to reproduce counts. When reloading floating-point CSVs, use
pd.read_csv(..., float_precision="round_trip"). Rounding a boundary threshold
can change decisions. An interval of thresholds can produce the same decisions
on discrete scores; the saved threshold is one deterministic representative,
not an intrinsically meaningful clinical cutoff.

## Interpretation and limitations

All points are descriptive development candidates, not clinically optimal,
clinically validated, or production thresholds. Although each model probability
is OOF, candidate thresholds are identified and described on the same pooled
OOF labels. Their reported performance is not an independent estimate after
threshold selection. Prior model and preprocessing choices also used development
data. No uncertainty intervals or future threshold stability are established.

PPV/NPV depend on prevalence (about 14.43% here). These are unweighted sample
estimates, not survey-weighted population inference. Random CV mixes cycles
and does not establish temporal generalization. High sensitivity requires many
false positives. Youden J equally weights sensitivity/specificity without
clinical consequence or resource assumptions; equality of sensitivity and
specificity is also only a statistical reference. The target describes diagnosed
diabetes, not prospective incident risk or undiagnosed-disease detection.

XGBoost has better observed specificity at matched 80% and 90% sensitivity:
404 and 643 fewer false positives than LR, corresponding to specificity gains
of about 1.85 and 2.94 percentage points. At 0.50 it has higher sensitivity
and PPV with nearly identical specificity. At their individual Youden points,
XGBoost has higher sensitivity but lower specificity than LR. These observations
support a better development profile at the requested matched-sensitivity points,
not universal threshold dominance or validated clinical benefit. No production
threshold is selected. No temporal validation, calibration modification,
SHAP/LIME, subgroup analysis, or risk-band definitions were performed.

## Execution and tests

```powershell
.venv/Scripts/python.exe -m scripts.analyze_nhanes_thresholds
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_thresholds_20261017
```

Use a new unused basetemp if Windows prevents reuse. The existing
test_nhanes.py integration tests are excluded because they read actual temporal
data. New tests inspect every pipeline's training/held-out indices, verify
shared folds and exactly one OOF prediction per participant/model, test valid
probabilities, compare tied-score counts with sklearn confusion matrices,
verify all rates using hand calculations, test exact/unattainable sensitivity
targets and deterministic tie rules, repeat synthetic OOF runs, validate result
schemas, enforce development-only reads, reject temporal labels, and verify
synthetic frozen-file hashes. Actual temporal data is never opened by these
tests or hashed. Production separately checks the real development-file hash.


## Recorded run

Development-only pytest: **35 passed in 31.83 seconds**. No production fit warnings.
Verified 51,160 OOF rows (25,580 per model), 1,962 curve rows, and 10 candidates.
All candidate confusion counts were independently reproduced from the saved OOF probabilities.
Development SHA-256 before/after: `f75e4db96bc672ea0f0f6b1913c484d4af56a9f0386541fca07cd22b5e4e15d9`.

| Model | Reference | Threshold (rounded) | Sensitivity | Specificity | PPV | NPV | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | Default 0.50 | 0.50000000 | 14.58% | 98.21% | 57.85% | 87.21% | 538 | 392 | 21498 | 3152 |
| LR | 80% sensitivity | 0.12480972 | 80.00% | 67.44% | 29.29% | 95.24% | 2952 | 7128 | 14762 | 738 |
| LR | 90% sensitivity | 0.07841132 | 90.00% | 53.29% | 24.52% | 96.93% | 3321 | 10224 | 11666 | 369 |
| LR | Youden J | 0.13264101 | 78.70% | 69.06% | 30.01% | 95.06% | 2904 | 6772 | 15118 | 786 |
| LR | Sensitivity ~= specificity | 0.15485607 | 73.36% | 73.36% | 31.71% | 94.23% | 2707 | 5831 | 16059 | 983 |
| XGBoost | Default 0.50 | 0.50000000 | 17.80% | 98.19% | 62.33% | 87.63% | 657 | 397 | 21493 | 3033 |
| XGBoost | 80% sensitivity | 0.14026810 | 80.00% | 69.28% | 30.51% | 95.36% | 2952 | 6724 | 15166 | 738 |
| XGBoost | 90% sensitivity | 0.08646801 | 90.00% | 56.23% | 25.74% | 97.09% | 3321 | 9581 | 12309 | 369 |
| XGBoost | Youden J | 0.12872706 | 82.52% | 67.03% | 29.67% | 95.79% | 3045 | 7218 | 14672 | 645 |
| XGBoost | Sensitivity ~= specificity | 0.16584124 | 74.39% | 74.39% | 32.87% | 94.51% | 2745 | 5607 | 16283 | 945 |

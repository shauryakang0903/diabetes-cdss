# Nested development-only hyperparameter tuning

## Input and fixed preprocessing

The sole cohort input is `data/processed/nhanes/nhanes_development_2011_2020.csv`
(25,580 adults; 3,690 positives and 21,890 negatives). The shared loader checks
the frozen counts, development cycle labels, unique participant IDs, adult ages,
and binary target. The target is physician-diagnosed diabetes from DIQ010.
No temporal-validation data is read, including for hashing. Development bytes
are hashed before and after the run; matching hashes are saved in provenance.
Previous preprocessing and baseline results are not overwritten.

Core-10 is unchanged: age, sex, race_ethnicity, bmi, waist_cm,
mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl, total_cholesterol_mg_dl,
and smoking_status. The seven numeric variables use median imputation; sex,
race/ethnicity, and smoking status use most-frequent imputation and one-hot
encoding with handle_unknown="ignore". Numeric scaling is applied only to LR.
All learned preprocessing stays inside a fresh Pipeline/ColumnTransformer.
No full-cohort preprocessing, weighting, or oversampling is performed.

## Nested design

Outer CV: StratifiedKFold, n_splits=5, shuffle=True, random_state=42.
Inner CV: StratifiedKFold, n_splits=3, shuffle=True, random_state=42,
applied separately within each outer-training fold. Both models receive the
same outer splits and the same relative inner splits within an outer fold.
No outer-test data is passed to a search, preprocessing fit, or early stopping.

Each candidate/inner-fold fit uses a cloned fresh full pipeline. Search scores
are mean inner-fold ROC-AUC. The winning candidate is refitted on that outer
training cohort only (`refit=True`) before predicting the untouched outer test
fold. No final model is refitted on all development data. Outer evaluation
estimates the performance of the specified search-and-refit procedure, rather
than the optimistically selected inner score. This protection against tuning
leakage does not guarantee an unbiased estimate of clinical or temporal utility.
Feature/preprocessing choices and previous comparisons already used development
data; this is not a fully untouched test of all research decisions.

## Prespecified spaces

LR: GridSearchCV over C = [0.01, 0.1, 1, 10, 100]. Fixed lbfgs with L2
(l1_ratio=0 under installed scikit-learn 1.9), max_iter=2000,
class_weight=None, random_state=42. Explicit deprecated penalty arguments are
avoided. No invalid solver/penalty combinations or solver search are introduced.

XGBoost: RandomizedSearchCV, n_iter=20, random_state=42. All distributions
are finite lists, sampled without replacement. The same 20 candidate
configurations are evaluated in each outer fold to avoid confounding parameter
stability with changes to the candidate pool.

| Parameter | Values |
|---|---|
| n_estimators | 100, 200, 300 |
| learning_rate | 0.03, 0.1, 0.3 |
| max_depth | 2, 3, 4, 6 |
| min_child_weight | 1, 5, 10 |
| subsample | 0.8, 1.0 |
| colsample_bytree | 0.8, 1.0 |
| reg_alpha | 0, 0.1, 1 |
| reg_lambda | 1, 5, 10 |

Other XGBoost settings inherit the untuned baseline: binary:logistic objective,
hist tree method on CPU, eval_metric=logloss, gamma=0, scale_pos_weight=1,
random_state=42, n_jobs=1. There is no eval_set, early stopping, or class/sample
weighting. Both searches run with n_jobs=1 and error_score="raise"; failed fits
cannot silently become selected or missing scores. There are 75 LR inner fits,
300 XGBoost inner fits, and ten outer-training refits in total.

## Outputs and interpretation

The `nested_tuning_outer_folds.csv` file records sizes, train/test prevalence,
best inner ROC-AUC, selected and complete estimator parameters, warnings,
package versions, seeds, features, and outer metrics. Average precision is the
PR summary (not trapezoidal PR-AUC). Brier score is mean squared probability
error. Accuracy, sensitivity, specificity, precision, and F1 use threshold 0.5
for description only. Specificity is TN/(TN+FP). No threshold is optimized.

The summary reports mean and sample SD (ddof=1) for all outer metrics.
Fold scores share overlapping training samples; SD is not a confidence interval
or an independent-sample standard error. The inner-search CSV records every
candidate's inner ROC-AUC mean, SD, and rank. Provenance records exact spaces,
iteration count, seeds, and development-file hashes. Dependencies are the same
as the pinned baseline environment; no new packages are required.

The hyperparameters CSV uses long format: selections per model/outer fold,
frequency counts for every LR setting (including zero counts), and number of
distinct selected values for each XGBoost parameter. A descriptive
`substantial_variation` flag is set if at least four of eight XGBoost parameters
have multiple selected values across folds. This predeclared heuristic is not
a statistical test; different configurations can yield similar performance.

Comparisons with the previous baseline reuse outer splits, enabling descriptive
paired comparisons. No clinical materiality margin or significance test is
specified. Lower Brier error does not isolate calibration quality. Random CV
mixes survey cycles; the scores are unweighted sample-level estimates, not
survey-weighted population inference or proof of temporal generalization.
The target is diagnosed diabetes, not prospective incident disease or detection
of undiagnosed diabetes. No final model selection, calibration, subgroup,
explanation, temporal evaluation, or threshold optimization is performed.

## Execution and tests

```powershell
.venv/Scripts/python.exe -m scripts.tune_nhanes_nested
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_nested_20261017
```

Use a new unused pytest basetemp for subsequent runs if Windows permissions on
an old directory prevent reuse. The pre-existing `test_nhanes.py` integration
tests are excluded because they read actual temporal-validation data. New tests
run small real searches, inspect every pipeline-fit row set against allowed
inner-training/outer-training rows, verify outer-test exclusion and shared
splits, validate parameter membership and schemas, repeat searches with seed 42,
reject temporal cycle labels, and verify synthetic frozen-file integrity. The
production runner separately verifies the actual development-file hash.

## Observed results

Development-only pytest result: **18 passed in 19.41 seconds**. No model/search
fit warnings were recorded in any of the ten outer fits. Before/after
development SHA-256 values match and also match the baseline experiment's hash.

| Outer metric, mean ± sample SD | Logistic Regression | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.809621 ± 0.009491 | 0.826579 ± 0.009334 |
| Average precision | 0.415944 ± 0.012120 | 0.451996 ± 0.015945 |
| Brier score | 0.102361 ± 0.001635 | 0.098840 ± 0.001710 |
| Accuracy | 0.861454 ± 0.001336 | 0.865911 ± 0.002822 |
| Sensitivity | 0.146612 ± 0.011750 | 0.178049 ± 0.016445 |
| Specificity | 0.981955 ± 0.001361 | 0.981864 ± 0.002212 |
| Precision | 0.577820 ± 0.017030 | 0.623401 ± 0.034899 |
| F1 | 0.233733 ± 0.015738 | 0.276762 ± 0.021814 |

| Outer fold | LR selected C | XGBoost selection |
|---|---:|---|
| 1 | 0.1 | Configuration X below |
| 2 | 0.1 | Configuration X below |
| 3 | 1 | Configuration X below |
| 4 | 0.1 | Configuration X below |
| 5 | 0.1 | Configuration X below |

Configuration X: n_estimators=200, learning_rate=0.03, max_depth=4,
min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
reg_alpha=1, reg_lambda=10. LR counts: C=0.1 selected four times, C=1 once;
C=0.01, 10, and 100 never selected. All eight XGBoost parameters have a single
selected value, so the substantial-variation flag is false. This stability is
conditional on the fixed sampled set of 20 configurations, not an assertion
that an exhaustive search or another seed would give the same answer.

Versus untuned baseline, LR ROC-AUC changes by +0.000032, average precision by
+0.000181, and Brier by -0.000012: negligible. XGBoost changes by +0.020533,
+0.045345, and -0.006712 respectively: a clear descriptive development-CV
improvement. Tuned XGBoost exceeds tuned LR by approximately 0.01696 ROC-AUC
and 0.03605 average precision, with Brier lower by 0.00352. These support an
observed development comparison advantage, not proven clinical materiality,
statistical significance, calibration superiority, or temporal generalization.
Sensitivity remains low at the deliberately unoptimized threshold of 0.5.

Several XGBoost selections lie at search boundaries (learning rate 0.03,
min_child_weight=10, alpha=1, lambda=10). The space was kept fixed; results
were not used to expand or rerun it. No final model was selected.

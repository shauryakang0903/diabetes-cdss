# Untuned development-only model comparison

Run from the project root:

```powershell
.venv/Scripts/python.exe -m scripts.compare_nhanes_baseline_models
```

The only cohort input is `data/processed/nhanes/nhanes_development_2011_2020.csv`:
25,580 adults, 3,690 positive and 21,890 negative outcomes. The shared loader
checks counts, unique IDs, development cycle labels, adult ages, and binary
outcomes. The target is physician-diagnosed diabetes from DIQ010. No temporal
cohort is loaded, summarized, or evaluated. Frozen input CSVs are not written.

## Fixed pipelines and parameters

The existing Core-10 features are unchanged. Numeric features (age, BMI, waist,
systolic BP, diastolic BP, HDL, and total cholesterol) use median imputation.
Sex, race/ethnicity, and smoking status use most-frequent imputation followed
by OneHotEncoder(handle_unknown="ignore"). Logistic Regression additionally
standardizes numeric features; XGBoost does not. All learned transformations
are inside Pipeline/ColumnTransformer and fitted only on training-fold rows.

Logistic Regression uses lbfgs, C=1, max_iter=2000, random_state=42,
class_weight=None, and the installed scikit-learn defaults for other parameters.
The larger iteration budget supports solver convergence and is not a searched
hyperparameter. It preserves the prior preprocessing probe configuration.

XGBoost uses binary:logistic, n_estimators=100, learning_rate=0.3, max_depth=6,
min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1, reg_alpha=0,
reg_lambda=1, scale_pos_weight=1, tree_method="hist", device="cpu",
eval_metric="logloss", random_state=42, and n_jobs=1. Other parameters retain
installed-version defaults. These fixed conventional baseline settings were
not searched. There is no eval_set or early stopping. XGBoost 3.4.1 was the
only newly installed package; dependency pins are in requirements-nhanes-baseline.txt.

Both models are unweighted, with no sample weights. Each fold CSV row records
all estimator get_params() values, including version defaults. XGBoost rows
also record the resolved booster configuration after fitting. Package versions,
feature list, preprocessing strategy, seed, threshold, and fold count are saved.
The provenance JSON records the development-file SHA-256 before and after the
experiment; no temporal file is read even to calculate a hash.

## Evaluation

StratifiedKFold uses five folds, shuffle=True, random_state=42. Splits are
generated once and reused for both models. Each fit uses a fresh full pipeline.
The model receives no held-out rows during training. There is no full-cohort
preprocessing or final full-cohort refit.

Primary metrics are ROC-AUC and average precision (the PR summary, not
trapezoidal PR-AUC). Brier score measures mean squared probability error;
lower is better. Accuracy, sensitivity, specificity, precision, and F1 use a
fixed descriptive 0.5 threshold. No clinical threshold is selected. Specificity
is TN/(TN+FP), computed from held-out predictions. Fold training and validation
sizes and positive rates are saved. Here "validation" in fold tables means
the held-out development fold, never the temporal cohort.

Summary CSVs report the mean and sample SD (ddof=1) of each metric. Overlapping
training samples make these fold scores dependent; fold SD is not a confidence
interval or an independent-sample standard error. Model fit warnings are both
surfaced and saved in the per-fold CSV. There is no tuning, nested CV,
calibration, threshold optimization, explanation analysis, subgroup analysis,
temporal evaluation, or final model selection.

## Tests and interpretation limits

```powershell
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_baseline_20261010
```

The excluded pre-existing data-engineering integration tests read temporal
validation. The tests exercised here cover both pipelines, missing values,
probability ranges, identical split membership, output schema and metrics,
development-only loading, rejection of temporal cycle labels, unchanged input
frames, and synthetic frozen-file sentinels. Actual development input integrity
is checked by the experiment's before/after hashes.

This is unweighted sample-level random CV, not survey-weighted population
inference. It mixes development cycles and does not establish temporal
generalization. The target describes diagnosed diabetes rather than prospective
incident risk or detection of undiagnosed diabetes. Brier differences describe
overall probability error, not calibration alone; no calibration analysis is
performed. No clinical materiality margin was prespecified, so observed metric
differences should not be presented as proven clinical benefit.

## Observed baseline results

| Metric (mean ± sample SD) | Logistic Regression | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.809590 ± 0.009452 | 0.806046 ± 0.006668 |
| Average precision | 0.415763 ± 0.012189 | 0.406650 ± 0.014604 |
| Brier score | 0.102373 ± 0.001648 | 0.105552 ± 0.002219 |
| Accuracy | 0.861572 ± 0.001300 | 0.858913 ± 0.003973 |
| Sensitivity | 0.148780 ± 0.011313 | 0.237127 ± 0.011215 |
| Specificity | 0.981727 ± 0.001389 | 0.963728 ± 0.004509 |
| Precision | 0.578406 ± 0.016545 | 0.525515 ± 0.033093 |
| F1 | 0.236538 ± 0.015042 | 0.326545 ± 0.013607 |

LR has a modest observed ranking advantage: ROC-AUC +0.003544 and average
precision +0.009113 versus XGBoost. Its Brier error is lower by 0.003179.
These are descriptive differences; clinical materiality and statistical
significance have not been established. XGBoost has higher sensitivity and
F1 at 0.5, with lower specificity and precision. Neither operating point is
selected for clinical use. No final model is selected.

There were no model-fit warnings in the recorded experiment. The first test
invocation encountered Windows permission errors in pytest's pre-existing
temporary directory/cache. The documented command uses a new project-local
temporary directory and disables pytest's cache; a new unused basetemp can be
chosen for another run. This environment issue did not change model settings.

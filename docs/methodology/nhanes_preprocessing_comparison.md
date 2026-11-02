# Development-only preprocessing probe

Run from the project root with:

```powershell
.venv/Scripts/python.exe -m scripts.compare_nhanes_preprocessing
```

The only dataset read by this harness is
`data/processed/nhanes/nhanes_development_2011_2020.csv`. Its cycle labels,
adult ages, binary outcomes, unique IDs, and frozen cohort counts are checked.
The harness never loads temporal validation. Frozen CSVs are not written.

## Fixed comparison

Both strategies use the existing Core-10 features, with age, BMI, waist,
systolic BP, diastolic BP, HDL, and total cholesterol treated as numeric;
sex, race/ethnicity, and smoking status are categorical.

- A_median: numeric median imputation.
- B_iterative: numeric IterativeImputer, default BayesianRidge estimator,
  max_iter=10, tol=0.001, sample_posterior=False, random_state=42.
- Both: most-frequent categorical imputation and OneHotEncoder with
  handle_unknown="ignore". Numeric StandardScaler is used for Logistic
  Regression; the reusable XGBoost preprocessing branch omits scaling.
- Both imputers retain all-missing columns (fallback zero) to preserve the
  feature schema. This is defensive behavior, not a clinical imputation rule.

Logistic Regression is only a probe: lbfgs, C=1, max_iter=2000, no class
weighting. There is no hyperparameter search. Using the observed class
distribution avoids introducing a different weighting objective into this
preprocessing comparison. Five identical stratified folds are shuffled using
seed 42. Every fold fits a fresh complete Pipeline on training rows only.
There is no preprocessing of the full cohort and no full-cohort model refit.

ROC-AUC, average precision, and Brier score use held-out probabilities.
Average precision is the PR summary, not trapezoidal PR-AUC. Accuracy, recall,
specificity, precision, and F1 use a fixed, untuned threshold of 0.5.
Specificity is TN/(TN+FP); Brier score compares probability with the binary
outcome. Mean and sample SD (ddof=1) describe the five folds. Their overlapping
training samples mean that SD is not an independent-sample standard error.

## Artifacts and reproducibility

The two CSVs under `results/nhanes/` contain per-fold scores and strategy
means/SDs. They include the seed, fold count, feature list, threshold, class
weight setting, and package versions. Per-fold fit warnings are preserved.
Dependencies used in this run are pinned in `requirements-nhanes-experiment.txt`.

Tests cover missing data, unseen categories, all-missing numeric columns,
Pipeline/ColumnTransformer structure, iterative imputation, training-only
statistics, identical fold membership, development-only loading, rejection
of validation labels, reproducibility, saved metrics/metadata, specificity,
and Brier arithmetic. Use:

```powershell
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py
```

The existing data-engineering integration tests in `test_nhanes.py` read the
temporal-validation cohort. They are intentionally excluded for this task to
honor the restriction on accessing that cohort; no existing code they test
was modified.

## Interpretation of the fixed run

The iterative strategy yielded modest mean improvements over median:
ROC-AUC +0.003285, average precision +0.007868, recall +0.010840,
and Brier score -0.000757. Specificity decreased by 0.000959.
All five iterative fits reached the 10-iteration limit without meeting the
early-stopping criterion. These warnings were not suppressed or used to tune
the experiment. The observed differences do not establish a material benefit:
no clinical materiality margin was prespecified and this single five-fold run
does not establish statistical significance. No strategy or final model is
selected here.

Recall is low at 0.5 for both strategies; this probe does not establish a useful
clinical operating threshold. Scores concern physician-diagnosed diabetes,
not detection of undiagnosed diabetes. This is an unweighted sample-level CV
comparison, not a survey-weighted population estimate. Random stratified CV
mixes development cycles and cannot establish temporal generalization.
Iterative imputation here is deterministic single imputation and does not
propagate imputation uncertainty. No XGBoost comparison, temporal evaluation,
calibration analysis, or explanation analysis was run.

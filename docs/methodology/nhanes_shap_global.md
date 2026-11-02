# Global SHAP explanation of the frozen Phase-2 model

## Scope and immutable model

This analysis explains the already-fitted nhanes_phase2_xgb_v1 artifact,
`models/nhanes/phase2_xgboost_pipeline.joblib`, using its metadata. The loader
verifies artifact SHA-256, fitted schema, model parameters, transformed names,
and recorded environment before explanation. Artifact, metadata, development
CSV, and previous result hashes are compared before and after the run. No
predictive model, imputer, encoder, calibration map, or threshold is fitted,
retuned, or changed. No new predictive model is compared.

The only cohort read is the frozen development CSV: 25,580 adults from NHANES
2011-March 2020 with 3,690 positive physician-diagnosed diabetes outcomes.
The completed 2021-2023 temporal cohort is not opened, inspected, or hashed,
and is not used as foreground or background for SHAP. Stored temporal result
files may be hashed to protect existing results, but their contents never enter
an explanation. Tests use the frozen artifact and small development samples;
other regression tests train only synthetic models in temporary directories.

## Full-cohort exact tree explanations

All 25,580 development rows are explained; there is no row sampling. Seed 42
is recorded and used for deterministic beeswarm jitter. The already-fitted
ColumnTransformer transforms the existing Core-10 predictors once, without
fit or fit_transform. Its exact 18 output names/order are compared with frozen
metadata and recorded in mapping, NPZ, and provenance. The original inputs are
age, sex, race_ethnicity, bmi, waist_cm, mean_systolic_bp, mean_diastolic_bp,
hdl_mg_dl, total_cholesterol_mg_dl, and smoking_status.

SHAP TreeExplainer is applied to the saved XGBoost booster with
feature_perturbation="tree_path_dependent", model_output="raw", and
approximate=False. No explicit background dataset is supplied. The reference
distribution uses frozen model training-path cover counts, not temporal rows
and not a newly selected background sample. Tree-path-dependent attribution
is a particular tree-based reference convention; it is not an interventional
causal estimate or a universal attribution independent of feature dependence.

Binary raw XGBoost margin is in **log-odds units** for diabetes=1. The base
value and SHAP values reconstruct this margin. SHAP values are not probability
changes, calibrated risks, regression coefficients, or clinical odds ratios.
The base value is a raw-margin reference, not the cohort prevalence. A positive
SHAP contribution raises the model margin relative to that reference; a negative
one lowers it. This describes the fitted model only.

Additivity is checked against booster.predict(DMatrix(transformed),
output_margin=True) on every explained row:

```text
base_value + sum(transformed_SHAP_values) ~= frozen_model_raw_margin
```

Both SHAP's own check and an independent comparison are enabled. The independent
tolerance is atol=1e-4 and rtol=1e-5. Maximum and mean absolute errors are saved.
Nonfinite values, unexpected binary SHAP shapes, inconsistent base values,
feature-name/shape mismatches, or failed additivity stop the analysis.

## Mapping and original-feature aggregation

The mapping is derived from fitted transformer output slices and encoder
categories, not by parsing underscore-delimited names. Seven numeric outputs
map one-to-one to their original feature. Eleven one-hot outputs map to sex,
race/ethnicity, and smoking status. All 18 indices must occur exactly once,
and all original Core-10 variables must be covered. Category values are saved
explicitly; numerical category labels on plots are NHANES codes, not ordinal
scores.

For participant i and original feature g, calculate:

```text
aggregated_SHAP[i,g] = sum(SHAP[i,j] for transformed column j belonging to g)
Core10_importance[g] = mean_i(abs(aggregated_SHAP[i,g]))
```

Signed contributions are summed **within each participant before taking the
absolute value**. This preserves opposing/cancelling one-hot contributions.
Simply summing the transformed columns' mean absolute SHAP would generally
overstate the aggregated magnitude. That alternative is recorded in explicitly
labelled NOT_aggregated columns for comparison, never used as primary importance.
This is an aggregation of existing transformed-feature attributions; it does
not rerun a different grouped-coalition SHAP game.

For every participant, summed Core-10 contributions must equal the total
transformed contributions, checked at atol=rtol=1e-10. The maximum aggregation
sum error is also recorded. Global tables report mean absolute, mean signed,
and population SD (ddof=0) across all explained participants. Mean signed SHAP
is not a universal directional effect. Ranks descend by mean absolute SHAP,
with fixed model-column/Core-10 order breaking ties deterministically. These
SDs describe heterogeneity of model contributions, not uncertainty intervals.

## Outputs and plots

Under results/nhanes/xai:

- shap_global_transformed_features.csv: 18 transformed-feature summaries.
- shap_global_core10.csv: 10 correctly aggregated original-feature summaries,
  plus the explicitly labelled alternative magnitude/rank diagnostics.
- shap_feature_mapping.csv: index, exact transformed name, original feature,
  category value when relevant, and readable plot label.
- shap_values_development.npz: transformed SHAP matrix, per-row base values,
  expected/base value, public SEQN identifiers, transformed names, Core-10
  aggregated values/names, and raw margins. No target labels or raw clinical
  predictors are stored. Load with np.load(..., allow_pickle=False).
- shap_global_provenance.json: artifact/model identity, hashes, versions,
  full-cohort scope, seed, background/reference convention, additivity,
  aggregation diagnostics, timestamps, and warnings.
- figures/shap_global_transformed_bar.png, shap_global_beeswarm.png,
  and shap_global_core10_bar.png: 300-DPI global figures, visually inspected.

Bar magnitudes are raw-margin/log-odds contributions. Beeswarm colors show
**frozen preprocessed values**, including imputed medians/modes and one-hot
indicators. Missingness is not itself a separate model feature: an imputed
measurement does not become an observed clinical measurement merely because
its color appears on a plot. All 18 features are shown and all development
participants contribute. Optional dependence/interaction plots are deliberately
omitted to keep this task focused on global importance.

## Interpretation limits

SHAP explains model behavior, not biological causality. Feature importance
does not establish clinical risk-factor effect size, a clinical odds ratio,
or the effect of changing a patient's measurement. Correlated predictors such
as BMI/waist can share or redistribute attribution under the model and chosen
reference convention. Encoded category importance is not a demographic fairness
assessment and no subgroup analysis is performed.

This is a global explanation on the training/development population, not a
new validation of discrimination, calibration, or transportability. Rankings
can depend on model structure, data distribution, preprocessing, correlated
features, and the SHAP reference convention. Prior observed temporal missingness
limitations remain; no temporal information is used to revise explanations,
features, or the predictive model. No LIME, calibration, threshold change,
risk-band definition, or new training is included.

## Environment, execution, and tests

SHAP 0.54.1 and Matplotlib 3.11.2 were added. The constraint file preserves
the frozen artifact's existing numpy, pandas, sklearn, XGBoost, scipy, and
joblib versions; no numerical/serialization package was upgraded or downgraded.
requirements-nhanes-xai.txt records the explanation dependencies and constraints.

```powershell
.venv/Scripts/python.exe -m scripts.explain_nhanes_shap_global
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_shap_all_20261018
```

Use a new unused basetemp if Windows prevents reuse. The existing raw-data
integration tests are excluded because they read actual temporal data. New
tests load the real frozen artifact and patch fitting methods to fail, verify
18 deterministic transformed names, complete mapping coverage, exact finite
SHAP shapes and additivity, cancellation-aware categorical aggregation,
deterministic ranks, saved output schemas/arrays/figures, and unchanged protected
hashes. A synthetic temporal sentinel checks preservation without opening the
actual temporal CSV. The production run separately verifies real artifact,
metadata, development and prior-result hashes.


## Recorded full-cohort run

**53 tests passed in 71.27 seconds.** No SHAP/XGBoost compatibility or plotting warnings were recorded.
Artifact SHA-256 verified and unchanged: `fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`.
All 25,580 development participants were explained, with 18 transformed and 10 original features.
Expected/base raw margin: -1.7905526161193848.
Maximum reconstruction error: 4.80790913571e-06; mean error: 7.22981089904e-07.
Maximum original/transformed sum discrepancy: 0.
The compressed NPZ occupies 3,310,679 bytes and loads without pickle.
All 27 earlier result files retain their hashes; no model or preprocessing fit occurred.

| Rank | Transformed feature | Mean absolute SHAP | Mean signed SHAP | SD |
|---:|---|---:|---:|---:|
| 1 | numeric__age | 0.920045 | -0.371287 | 1.075246 |
| 2 | numeric__waist_cm | 0.344519 | -0.119913 | 0.444620 |
| 3 | numeric__total_cholesterol_mg_dl | 0.258703 | -0.033691 | 0.319449 |
| 4 | categorical__race_ethnicity_3.0 | 0.196387 | -0.008900 | 0.218740 |
| 5 | numeric__hdl_mg_dl | 0.179950 | -0.034479 | 0.220995 |
| 6 | numeric__mean_systolic_bp | 0.134049 | -0.034373 | 0.153748 |
| 7 | numeric__mean_diastolic_bp | 0.110400 | 0.000477 | 0.144383 |
| 8 | numeric__bmi | 0.102575 | -0.022589 | 0.122355 |
| 9 | categorical__sex_1.0 | 0.031957 | 0.001594 | 0.036865 |
| 10 | categorical__race_ethnicity_1.0 | 0.031657 | -0.001624 | 0.047806 |
| 11 | categorical__race_ethnicity_6.0 | 0.019068 | 0.000807 | 0.031663 |
| 12 | categorical__race_ethnicity_2.0 | 0.008142 | 0.001028 | 0.012561 |
| 13 | categorical__smoking_status_1.0 | 0.005964 | -0.000866 | 0.007038 |
| 14 | categorical__smoking_status_2.0 | 0.005196 | -0.000406 | 0.008784 |
| 15 | categorical__sex_2.0 | 0.005060 | 0.000491 | 0.006160 |
| 16 | categorical__race_ethnicity_4.0 | 0.003215 | -0.000313 | 0.005138 |
| 17 | categorical__smoking_status_0.0 | 0.001406 | 0.000037 | 0.002134 |
| 18 | categorical__race_ethnicity_7.0 | 0.000000 | 0.000000 | 0.000000 |

| Rank | Original Core-10 feature | Mean absolute aggregated SHAP | Mean signed aggregated SHAP | SD |
|---:|---|---:|---:|---:|
| 1 | age | 0.920045 | -0.371287 | 1.075246 |
| 2 | waist_cm | 0.344519 | -0.119913 | 0.444620 |
| 3 | total_cholesterol_mg_dl | 0.258703 | -0.033691 | 0.319449 |
| 4 | race_ethnicity | 0.217174 | -0.009002 | 0.245335 |
| 5 | hdl_mg_dl | 0.179950 | -0.034479 | 0.220995 |
| 6 | mean_systolic_bp | 0.134049 | -0.034373 | 0.153748 |
| 7 | mean_diastolic_bp | 0.110400 | 0.000477 | 0.144383 |
| 8 | bmi | 0.102575 | -0.022589 | 0.122355 |
| 9 | sex | 0.036995 | 0.002085 | 0.042672 |
| 10 | smoking_status | 0.009081 | -0.001236 | 0.012796 |

Correct categorical aggregation versus the deliberately labelled shortcut:

| Original feature | Correct aggregated mean absolute | Sum of column mean absolutes | Magnitude reduction | Correct rank | Shortcut rank |
|---|---:|---:|---:|---:|---:|
| race_ethnicity | 0.217174 | 0.258469 | 15.98% | 4 | 4 |
| sex | 0.036995 | 0.037016 | 0.06% | 9 | 9 |
| smoking_status | 0.009081 | 0.012566 | 27.74% | 10 | 10 |

The Core-10 order is unchanged in this run, but magnitude reductions for race/ethnicity and smoking status are nontrivial. This does not justify using the shortcut: per-participant signed aggregation remains the required calculation.
Age, waist circumference, and total cholesterol lead both numeric/transformed and original-feature importance. These are model-attribution magnitudes in log-odds units, not clinical risk-factor effects.

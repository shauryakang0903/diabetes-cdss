# Frozen Phase-2 XGBoost artifact

## Why the artifact is frozen

The Phase-2 predictive specification is fixed after development-only model,
preprocessing, calibration, threshold, and leave-cycle-out analyses and the
completed primary temporal evaluation. The artifact provides a single fitted
pipeline and auditable metadata for later explainability and inference. This
stage is not another model comparison or optimization experiment.

The 2021-2023 cohort has already been evaluated. It cannot be reused for further
fitting, preprocessing estimates, tuning, calibration, threshold selection, or
artifact construction while continuing to call its prior evaluation untouched.
This build never opens that cohort, even for a hash or a serialization probe.
Previously saved temporal metrics/completeness tables are copied into metadata
for documentation only; they do not influence any fitted parameter.

## Training data and target

The sole fitting input is
`data/processed/nhanes/nhanes_development_2011_2020.csv`: adults aged at least
20 from NHANES 2011-2012, 2013-2014, 2015-2016, and 2017-March 2020, with a
valid binary physician-diagnosed diabetes outcome. N=25,580, positives=3,690,
negatives=21,890. The existing development loader checks frozen counts,
development cycle labels, unique IDs, adult ages, and binary outcomes.

Target `diabetes` means 1=physician-diagnosed diabetes from DIQ010 and 0=no
physician-diagnosed diabetes. This is not prospective incident disease risk or
an undiagnosed-diabetes detection endpoint. A model probability is not a clinical
diagnosis, treatment recommendation, or demonstration of clinical utility.

## Exact pipeline

Inputs, in the recorded Core-10 order: age, sex, race_ethnicity, bmi, waist_cm,
mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl, total_cholesterol_mg_dl,
smoking_status. Numeric features are age, BMI, waist, systolic/diastolic BP,
HDL, and total cholesterol. Categorical features are sex, race/ethnicity,
and smoking status.

- Numeric SimpleImputer(strategy="median", keep_empty_features=True).
- Categorical SimpleImputer(strategy="most_frequent", keep_empty_features=True).
- OneHotEncoder(handle_unknown="ignore", sparse_output=False).
- No numeric standardization.
- All learned transformations remain inside the serialized ColumnTransformer
  and Pipeline. Fitted medians, modes, categories, and transformed feature names
  are also documented in metadata for auditing.

XGBClassifier: n_estimators=200, learning_rate=0.03, max_depth=4,
min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=1,
reg_lambda=10, objective="binary:logistic", scale_pos_weight=1, gamma=0,
tree_method="hist", device="cpu", eval_metric="logloss", n_jobs=1,
random_state=42. There is no class/sample weighting, post-hoc calibration,
early stopping, search, or additional model comparison. Complete installed-version
estimator parameters and the resolved booster configuration are stored in JSON.

The complete pipeline is fitted once on all development rows. It is saved as
`models/nhanes/phase2_xgboost_pipeline.joblib`, model version
`nhanes_phase2_xgb_v1`. Metadata is
`models/nhanes/phase2_xgboost_metadata.json`. The build refuses to overwrite an
existing artifact or metadata file. A future explicitly authorized model version
requires a separate versioned location; v1 must be preserved.

## Frozen reference operating points

Exact Python-float values are loaded with round-trip CSV precision from
`results/nhanes/threshold_candidate_points.csv`, never reselected or rounded.
Four names are supported: default, sensitivity_80_reference,
sensitivity_90_reference, and youden_reference. They are development-derived
operating references, **not clinically validated cutoffs** or a chosen production
clinical threshold. Classification uses p >= the named saved threshold.
No Low/Moderate/High clinical risk bands are defined.

## Serialization and provenance

The entire fitted preprocessing/classifier pipeline is serialized with
joblib.dump(compress=3), not just the raw booster. Development-derived probe
rows plus one artificial all-missing row are predicted before saving and after
reloading. No labels/performance metrics or temporal participants are used in
that check. Predictions must match at rtol=1e-7 and atol=1e-8. No model is
refitted during reload checks.

Metadata records cohort/target definitions, class counts, feature schemas,
fitted preprocessing summaries, all model parameters, reference thresholds,
artifact SHA-256, development/threshold-result hashes, source-code hashes,
timestamps, random seed, Python and package versions, temporal documentation
sources/hashes, serialization equivalence, and warnings. Both frozen input
files and all prior result files are checked for unchanged hashes after the
build. Actual temporal cohort data is never read or hashed. NaN model-parameter
sentinels are represented as the string "NaN" for strict JSON; actual estimator
settings remain unchanged.

Joblib uses pickle. Load only trusted artifacts: a checksum detects corruption
or inconsistency but is not authentication of a malicious artifact/metadata pair.
Use the recorded Python/library environment; sklearn/XGBoost pickle compatibility
across versions is not guaranteed. The loader warns on recorded-package or
Python major/minor differences and verifies artifact SHA-256, v1 metadata,
Core-10 schema, transformed feature names, estimator type, class order [0,1],
and frozen model parameters. It does not silently upgrade or refit an artifact.

The metadata cannot contain its own self-referential hash; it contains the
artifact hash and input/source hashes. The artifact and its metadata should be
distributed and archived together. Later explainability must load this exact
pipeline and recorded transformed-feature schema instead of recreating a model
from parameters or reconstructing a new encoder.

## Loading and prediction

```python
from src.models.nhanes_phase2_model import (
    load_frozen_model, load_metadata, predict_probabilities, predict_at_reference,
)

model = load_frozen_model()
metadata = load_metadata()
probability = predict_probabilities(samples, model)
reference_decision = predict_at_reference(samples, "sensitivity_80_reference")
```

`samples` must be a DataFrame with harmonized Core-10 names/units and numeric
NHANES category codes. This utility does not derive features from raw NHANES
questionnaires. Extra fields such as SEQN and target are ignored, and expected
names/order are preserved. Missing values, including an entirely missing
predictor row, are allowed because saved imputers handle them. Missing columns,
duplicate names, empty frames, free-text codes, and infinite values produce
clear errors. Numeric-string coercion is input type validation only; no learned
preprocessing occurs outside the saved pipeline. Unknown numeric categorical
codes retain the frozen encoder's handle_unknown="ignore" behavior rather than
triggering a new encoding strategy. Callers remain responsible for harmonized
units and coding.

Prediction functions return positive-class probabilities or binary reference
decisions, with no fitting. To relocate the pair, pass artifact_path and
metadata_path explicitly to the loader/reference helper. A previously loaded
pipeline can be passed to predict_probabilities to avoid repeated disk loading.

## Temporal summary and known limitations

Existing temporal result files are the sole documentation source. The completed
primary XGBoost evaluation had ROC-AUC approximately 0.81235, average precision
0.42473, Brier 0.10001, log loss 0.32523, calibration slope 1.09186, and joint
calibration intercept 0.13514. Exact saved values and their source hash are
stored in metadata, not recomputed in this task. Intercept/slope are evaluation
diagnostics, not applied probability corrections.

Temporal missingness was substantially higher than development missingness.
Earlier XGBoost ROC-AUC was about 0.83141 in complete Core-10 cases versus
0.77123 in participants missing at least one predictor. Metadata documents
the exact previously saved completeness-group summary and source hash. These
are descriptive robustness observations, not evidence of a causal missingness
effect or permission to alter preprocessing. Sensitivity-reference transport
did not establish clinical acceptability, and no final clinical cutoff has
been selected. Unweighted NHANES sample results do not establish population-wide
performance, prospective utility, or generalization to unrelated health systems.

No new temporal predictions/evaluation, SHAP/LIME, subgroup analysis, model
comparison, recalibration, threshold revision, or UI is implemented here.

## Build and tests

```powershell
.venv/Scripts/python.exe -m scripts.freeze_nhanes_phase2_model
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_freeze_all_20261018
```

The build command is intentionally one-time for a given output location; do not
rerun it to overwrite v1. Use a new unused pytest basetemp if Windows prevents
reuse. Existing raw-data integration tests are excluded because they reload
the actual temporal cohort; the processed-data/model tests use synthetic cohorts
and do not refit the real production artifact. New tests check build/reload,
probability equivalence and ranges, missing values/required names, frozen
thresholds/parameters, metadata provenance, development-only fit/read scope,
unchanged input hashes, overwrite prevention, checksum failure, and version
warnings. The production probe separately verifies the actual saved artifact.


## Recorded v1 build

**49 tests passed in 88.35 seconds.** No production training or load warnings.
Fitted development N=25580, positive=3690, negative=21890.
Feature count: 10 before preprocessing, 18 afterward.
Training UTC: 2026-10-18T09:10:39.774804+00:00 to 2026-10-18T09:10:43.367082+00:00.
Artifact size: 85322 bytes.

Artifact SHA-256: `fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`.

Reload equivalence: passed on 129 development-derived/artificial probe rows; maximum absolute probability difference 0.0.
All 27 previous result files and all fitting/documentation inputs retain their hashes.
The temporal cohort itself was not accessed during artifact creation or verification.

| Reference name | Exact stored threshold |
|---|---:|
| default | 0.5 |
| sensitivity_80_reference | 0.14026810228824615 |
| sensitivity_90_reference | 0.08646801114082336 |
| youden_reference | 0.12872706353664398 |

Recorded Python version: `3.13.4`.

| Package | Version |
|---|---|
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| joblib | 1.6.0 |
| scipy | 1.18.1 |

Previously saved temporal metrics, copied exactly for documentation:

| Metric | Saved value |
|---|---:|
| roc_auc | 0.812354197802452 |
| average_precision | 0.4247273958297281 |
| brier_score | 0.10001399529978304 |
| log_loss | 0.3252287354709952 |
| calibration_slope | 1.0918583561570374 |
| calibration_intercept | 0.13513768635391243 |
| ece | 0.006950486976801138 |

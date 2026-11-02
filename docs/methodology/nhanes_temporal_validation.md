# First primary frozen temporal evaluation

## Independence and fixed decisions

This is the first primary evaluation of NHANES 2021-2023 using decisions fixed
after development on 2011-March 2020. It was explicitly authorized on
2026-10-18. The only cohort inputs are the two frozen processed CSVs:

- data/processed/nhanes/nhanes_development_2011_2020.csv: 25,580 adults,
  3,690 positive and 21,890 negative outcomes.
- data/processed/nhanes/nhanes_validation_2021_2023.csv: 7,536 adults,
  1,070 positive and 6,466 negative outcomes.

Before any model fit, the runner verifies Core-10 and target columns, valid
adult/binary outcomes, within-cohort ID uniqueness, correct cycle labels,
frozen counts, and zero SEQN overlap. It records sizes, prevalence, and per-feature
missing counts/fractions in a preflight provenance JSON. The JSON is written
with status preflight_complete before either model is fitted. Missingness is
described only; it never changes preprocessing or model decisions.

Hashes of both cohort files and all development-reference files are checked
before/after execution. Existing result files are independently hashed to verify
that they were preserved. Provenance records UTC timestamps, versions, full
model parameters, thresholds at full precision, preflight observations, warnings,
and the completed status. A model-parameter NaN sentinel is encoded as string
"NaN" for strict JSON, not changed in the actual estimator. No raw XPTs are read
or changed by the primary evaluation. No additional packages were required.

## Frozen pipelines fitted once

The same Core-10 features are used: age, sex, race_ethnicity, bmi, waist_cm,
mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl, total_cholesterol_mg_dl,
smoking_status. Seven numeric fields use median imputation, with StandardScaler
only for LR. Sex, race/ethnicity, and smoking status use most-frequent imputation
and OneHotEncoder(handle_unknown="ignore"). All preprocessing remains inside
the complete sklearn-compatible pipeline.

LR is fixed at L2 (l1_ratio=0 in installed sklearn 1.9), lbfgs, C=0.1,
max_iter=2000, class_weight=None, seed 42. XGBoost is fixed at 200 trees,
learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8,
colsample_bytree=0.8, reg_alpha=1, reg_lambda=10, binary:logistic objective,
gamma=0, scale_pos_weight=1, CPU hist tree method, eval_metric=logloss,
n_jobs=1, seed 42, with no early stopping. No weighting or post-hoc calibration
is used. Other defaults are recorded in provenance.

Each pipeline is fitted exactly once on the full development cohort, then
applied directly to all temporal participants. Validation rows never enter an
estimator/preprocessor .fit call. There is no CV, tuning, feature revision,
imputation adaptation, recalibration, threshold search, or selection of a model
using temporal labels. The already-generated predictions are also used for the
data-completeness descriptions; groups do not trigger refitting or reprediction.
No final clinical operating rule or risk bands are defined.

## Metrics and calibration conventions

ROC-AUC, average precision (not trapezoidal PR-AUC), Brier score, and log loss
describe the temporal predictions. Float64 log loss clips probabilities to
machine epsilon and 1-epsilon; other probability metrics use original outputs.
Calibration diagnostics reuse the development experiment's functions unchanged:

- Joint unpenalized logistic diagnostic:
  logit P(Y=1) = intercept + slope * logit(prediction). Both coefficients are
  estimated together, with ideal values 0 and 1. This is not the offset-based
  calibration-in-the-large intercept. Only diagnostic logits use clipping to
  [1e-6, 1-1e-6]. The diagnostic optimizes a likelihood using test labels solely
  to measure calibration; its coefficients are never applied to predictions.
  It does not fit a new predictive pipeline or deployable calibrator.
- ECE uses ten fixed equal-width bins, left-inclusive/right-exclusive, last bin
  including p=1: sum(n_bin/N * abs(mean_prediction_bin - observed_rate_bin)).
  ECE is supportive and bin-dependent. Empty bins are retained with missing means.
- Mean predicted probability and observed prevalence are recorded separately.
  Their difference is an aggregate descriptive diagnostic.

The calibration-curve CSV contains ten points per model, with bin edges,
participant counts, mean prediction, and observed event fraction. Sparse tail
bins should not be interpreted as precise estimates. No probability model is
altered based on these diagnostics.

## Frozen operating points and development comparisons

Thresholds are loaded from results/nhanes/threshold_candidate_points.csv using
float_precision="round_trip". Default 0.50 and the saved 80%, 90%, and Youden
references are applied unchanged to both models; the equality reference is not
part of this primary evaluation. No temporal-optimal cutoff is computed.
Prediction rule is p >= saved threshold. Full numeric precision is retained.

Threshold outputs include sensitivity, specificity, PPV, NPV, accuracy,
balanced accuracy, F1, TP/FP/TN/FN, predicted-positive fraction, FPR/FNR,
and Youden J. Undefined PPV/NPV are missing, not assigned zero. Intended target
sensitivity and signed/absolute deviations are recorded for 80% and 90%
references; they are absent for default/Youden.

Temporal threshold metrics are compared with pooled development OOF metrics.
Signed differences for all requested rates are saved. For sensitivity and
specificity, saved leave-one-cycle-out minimum/maximum values are reported,
together with temporal-minus-each-bound differences, below/within/above status,
and absolute distance outside that observed cycle range. An observed four-cycle
range is descriptive, not a confidence or prediction interval.

Threshold-free temporal metrics are compared against nested tuning mean
ROC-AUC/AP/Brier. Signed and magnitude differences are both saved. Calibration
diagnostics and log loss are compared with raw-model development calibration
OOF means under the same diagnostic definitions. The raw calibration experiment
uses the exact fixed configurations. The nested LR comparison has an important
qualification: one outer fold previously selected C=1 instead of C=0.1; the
temporal LR is always C=0.1. Nested XGBoost settings were identical across folds.

Development references are fold means or pooled OOF estimates, as identified in
column names. The temporal predictions come from a model trained on all
development participants. Training size, case mix, prevalence, calendar period,
missingness, and measurement protocols differ. Differences are descriptive;
no formal hypothesis test, significance claim, or arbitrary acceptability margin
is introduced. No comparison is used to revise the frozen decisions.

## Data-completeness descriptions

Before imputation, temporal participants are divided into all Core-10 fields
observed versus one or more missing. N, positive/negative counts, prevalence,
ROC-AUC when both classes exist, AP, and Brier are reported using the same
full-cohort predictions. Empty groups produce missing metrics. Single-class
groups have no ROC-AUC; AP is missing if there are no positives and equals one
for an all-positive nonempty group. Brier remains defined for a nonempty group.
Missing-field counts 0, 1, 2, and 3+ are recorded in preflight provenance.

This is descriptive robustness by completeness, not demographic fairness
analysis. Complete and incomplete participants can differ in outcome prevalence
and case mix; differences cannot establish that missingness caused a performance
change. No model is adapted, selected, or retrained using these groups.

## Tests and execution

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_nhanes_temporal
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_temporal_all_20261018
```

New tests verify overlap/schema checks, frozen counts and paths, before-fit
preflight recording, pipeline/preprocessing fits receiving development rows only,
one fit per model, fixed estimator parameters, saved threshold loading, no tuning
or calibration/threshold-selection calls, valid probabilities, metric arithmetic,
completeness edge cases, comparison fields, result schemas, and frozen-input/
prior-result hashes. Synthetic cohorts are used for fitting tests. Existing
raw-data engineering integration tests in test_nhanes.py are excluded because
the primary task concerns frozen processed inputs and does not require raw-XPT
reloading. All remaining project tests are run. No actual temporal model is
refitted for tests or for reporting.

## Limits on interpretation

This is a single later NHANES cohort, not independent validation in an unrelated
health system or a prospective clinical utility study. Results are unweighted
sample-level estimates rather than survey-weighted national estimates. AP,
PPV/NPV, and Brier depend on prevalence/case mix. Calibration line coefficients
alone do not establish reliable probabilities throughout the score range.
Sensitivity targets derived from development are not guaranteed to transport;
there is no prespecified clinical acceptance margin in this request. Temporal
data has now been evaluated and can no longer be considered untouched for future
development choices. Any subsequent adaptation must be explicitly separated from
this frozen primary evaluation and independently assessed later.

The target is physician-diagnosed diabetes, not incident future disease or
detection of undiagnosed diabetes. No change of model, recalibration, training
on 2021-2023, threshold revision, SHAP/LIME, demographic subgroup/fairness
analysis, or risk-band definition is performed.


## Recorded primary evaluation

**44 tests passed in 85.74 seconds.** No production fit or calibration-diagnostic warnings.
Both real cohort hashes and all development-reference hashes are unchanged. All 22 prior result files were preserved.
The primary run fitted each frozen pipeline once; subsequent reporting used saved results only.

| Cohort | N | Positive | Negative | Prevalence |
|---|---:|---:|---:|---:|
| development | 25580 | 3690 | 21890 | 14.425% |
| temporal_validation | 7536 | 1070 | 6466 | 14.199% |

Zero SEQN overlap was verified. Missingness before imputation:

| Core-10 feature | Development missing N | Development missing % | Temporal missing N | Temporal missing % |
|---|---:|---:|---:|---:|
| age | 0 | 0.00% | 0 | 0.00% |
| sex | 0 | 0.00% | 0 | 0.00% |
| race_ethnicity | 0 | 0.00% | 0 | 0.00% |
| bmi | 1695 | 6.63% | 1784 | 23.67% |
| waist_cm | 2788 | 10.90% | 1983 | 26.31% |
| mean_systolic_bp | 2778 | 10.86% | 1885 | 25.01% |
| mean_diastolic_bp | 2778 | 10.86% | 1885 | 25.01% |
| hdl_mg_dl | 2884 | 11.27% | 2234 | 29.64% |
| total_cholesterol_mg_dl | 2884 | 11.27% | 2234 | 29.64% |
| smoking_status | 24 | 0.09% | 27 | 0.36% |

Temporal missing-field counts: 0: 4969 (65.94%), 1: 146 (1.94%), 2: 574 (7.62%), 3+: 1847 (24.51%).

Threshold-free metrics and calibration diagnostics:

| model | roc_auc | average_precision | brier_score | log_loss |
|---|---|---|---|---|
| LR | 0.794153 | 0.378308 | 0.103945 | 0.337565 |
| XGBoost | 0.812354 | 0.424727 | 0.100014 | 0.325229 |

| model | calibration_intercept | calibration_slope | ece | mean_predicted_probability | observed_prevalence |
|---|---|---|---|---|---|
| LR | 0.065766 | 1.014180 | 0.008024 | 0.137393 | 0.141985 |
| XGBoost | 0.135138 | 1.091858 | 0.006950 | 0.141120 | 0.141985 |

Signed temporal minus development differences:

| model | difference_vs_nested_roc_auc | difference_vs_nested_average_precision | difference_vs_nested_brier_score | difference_vs_raw_oof_log_loss |
|---|---|---|---|---|
| LR | -0.015469 | -0.037636 | 0.001584 | 0.005605 |
| XGBoost | -0.014225 | -0.027268 | 0.001174 | 0.005013 |

| model | difference_vs_raw_oof_calibration_intercept | difference_vs_raw_oof_calibration_slope | difference_vs_raw_oof_ece |
|---|---|---|---|
| LR | 0.057839 | 0.008618 | -0.002773 |
| XGBoost | 0.052785 | 0.031475 | -0.003653 |

Frozen threshold rates (full-precision values remain in the CSV; printed thresholds are rounded):

| model | operating_point | threshold | sensitivity | specificity | precision | negative_predictive_value |
|---|---|---|---|---|---|---|
| LR | Default 0.50 | 0.500000 | 0.092523 | 0.986236 | 0.526596 | 0.867855 |
| LR | Frozen 80% reference | 0.124810 | 0.771028 | 0.658831 | 0.272187 | 0.945616 |
| LR | Frozen 90% reference | 0.078411 | 0.893458 | 0.509279 | 0.231533 | 0.966539 |
| LR | Frozen Youden reference | 0.132641 | 0.751402 | 0.679091 | 0.279264 | 0.942882 |
| XGBoost | Default 0.50 | 0.500000 | 0.143925 | 0.984380 | 0.603922 | 0.874193 |
| XGBoost | Frozen 80% reference | 0.140268 | 0.792523 | 0.664244 | 0.280888 | 0.950852 |
| XGBoost | Frozen 90% reference | 0.086468 | 0.899065 | 0.521652 | 0.237238 | 0.968974 |
| XGBoost | Frozen Youden reference | 0.128727 | 0.814953 | 0.640272 | 0.272670 | 0.954357 |

| model | operating_point | tp | fp | tn | fn | accuracy | balanced_accuracy | f1 | predicted_positive_fraction |
|---|---|---|---|---|---|---|---|---|---|
| LR | Default 0.50 | 99 | 89 | 6377 | 971 | 0.859342 | 0.539380 | 0.157393 | 0.024947 |
| LR | Frozen 80% reference | 825 | 2206 | 4260 | 245 | 0.674761 | 0.714929 | 0.402341 | 0.402203 |
| LR | Frozen 90% reference | 956 | 3173 | 3293 | 114 | 0.563827 | 0.701369 | 0.367763 | 0.547903 |
| LR | Frozen Youden reference | 804 | 2075 | 4391 | 266 | 0.689358 | 0.715246 | 0.407192 | 0.382033 |
| XGBoost | Default 0.50 | 154 | 101 | 6365 | 916 | 0.865048 | 0.564153 | 0.232453 | 0.033838 |
| XGBoost | Frozen 80% reference | 848 | 2171 | 4295 | 222 | 0.682458 | 0.728384 | 0.414771 | 0.400610 |
| XGBoost | Frozen 90% reference | 962 | 3093 | 3373 | 108 | 0.575239 | 0.710359 | 0.375415 | 0.538084 |
| XGBoost | Frozen Youden reference | 872 | 2326 | 4140 | 198 | 0.665074 | 0.727613 | 0.408622 | 0.424363 |

Target-sensitivity transport, with signed temporal minus development differences:

| model | operating_point | sensitivity | absolute_sensitivity_target_deviation | difference_vs_development_oof_sensitivity | difference_vs_development_oof_specificity | cycle_sensitivity_min | cycle_sensitivity_max | distance_outside_cycle_sensitivity_range | distance_outside_cycle_specificity_range |
|---|---|---|---|---|---|---|---|---|---|
| LR | Frozen 80% reference | 0.771028 | 0.028972 | -0.028972 | -0.015541 | 0.762784 | 0.821831 | 0.000000 | 0.000000 |
| LR | Frozen 90% reference | 0.893458 | 0.006542 | -0.006542 | -0.023658 | 0.879261 | 0.915493 | 0.000000 | 0.000000 |
| XGBoost | Frozen 80% reference | 0.792523 | 0.007477 | -0.007477 | -0.028584 | 0.776989 | 0.810427 | 0.000000 | 0.016074 |
| XGBoost | Frozen 90% reference | 0.899065 | 0.000935 | -0.000935 | -0.040660 | 0.883523 | 0.907042 | 0.000000 | 0.013007 |

Complete-case and missing-feature descriptions:

| model | completeness_group | n | positive_n | prevalence | roc_auc | average_precision | brier_score |
|---|---|---|---|---|---|---|---|
| LR | all_core10_observed | 4969 | 699 | 0.140672 | 0.811778 | 0.404788 | 0.100142 |
| LR | one_or_more_core10_missing | 2567 | 371 | 0.144527 | 0.756987 | 0.318378 | 0.111307 |
| XGBoost | all_core10_observed | 4969 | 699 | 0.140672 | 0.831413 | 0.451840 | 0.095843 |
| XGBoost | one_or_more_core10_missing | 2567 | 371 | 0.144527 | 0.771226 | 0.373783 | 0.108088 |

### Descriptive interpretation

XGBoost retained higher AUC/AP and lower Brier/log loss than LR. Its AUC dropped
by 0.01422 and AP by 0.02727 versus nested development CV; LR drops were 0.01547
and 0.03764. Both temporal AUCs are below the minimum observed leave-cycle-out
AUC (LR 0.80367; XGBoost 0.81957). Discrimination therefore generalized partially,
with a measurable descriptive decline rather than preservation of the development
point estimates. No significance or clinical-materiality claim is made.

Aggregate calibration diagnostics were broadly preserved without recalibration,
but are not perfect: XGBoost slope was 1.09186 versus development 1.06038 and
joint intercept 0.13514 versus 0.08235. Mean XGBoost prediction was 14.112%, near
14.199% observed prevalence; LR mean was 13.739%. ECE was lower than development
for both, but Brier/log loss worsened. Low ECE or a near-ideal global line does
not establish reliable individual probabilities. The XGBoost 0.4-0.5 bin had
mean prediction 0.445 versus observed rate 0.525 (N=217); high-probability tail
bins are sparse (e.g. only 19 XGBoost participants in 0.7-0.8). These are
observations only and were not used to adapt the probabilities.

XGBoost achieved sensitivity 79.252% at the frozen 80% reference and 89.907%
at the 90% reference: shortfalls 0.748 and 0.093 percentage points. Both are
within previous cycle sensitivity ranges, but specificity fell to 66.424% and
52.165%. These are 2.858 and 4.066 percentage points below pooled development
OOF specificity, and 1.607 and 1.301 points below the previous cycle minima.
Predicted-positive fractions increased from 37.826% to 40.061% and from 50.438%
to 53.808%. Thus sensitivity targets transported closely while false-positive
burden increased. No prespecified clinical acceptance margins exist, so this
cannot establish that threshold transport is clinically acceptable. Default
0.50 sensitivity was especially low (LR 9.252%; XGBoost 14.393%).

Missingness increased substantially, especially BMI/waist/BP/lipids. One or
more fields were missing for 2,567 participants (34.06%); 1,847 had three or
more missing fields. XGBoost AUC was 0.83141 in complete cases versus 0.77123
with missing fields, with Brier 0.09584 versus 0.10809. LR showed the same
direction. Prevalence was similar but not identical in those groups. This
supports a descriptive completeness-related robustness concern; it does not
prove missingness caused the degradation or justify changing preprocessing.
Case mix, measurement and participation changes may also contribute.

No model, probability calibration, threshold, feature, or preprocessing setting
was changed in response to this evaluation. The primary evaluation is now
complete; temporal results must not be retroactively treated as untouched in
future development.

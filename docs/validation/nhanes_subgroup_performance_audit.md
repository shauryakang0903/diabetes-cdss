# Frozen NHANES XGBoost subgroup performance audit

**POST-HOC / EXPLORATORY SUBGROUP AUDIT — not fairness certification.**

## Purpose and scope

This transparency analysis describes performance variation of the existing
`nhanes_phase2_xgb_v1` diabetes classifier across demographic, clinical and
data-completeness subgroups. It does not train, select, recalibrate or replace a
model. It does not change features, probability thresholds, risk boundaries,
recommendation rules or the application.

The 2021–2023 cohort was evaluated previously. Applying the frozen artifact again
here is explicitly a **post-hoc/exploratory subgroup analysis**, not untouched
validation or an independent confirmatory study. Results should generate questions
for subsequent evaluation, not certification of fair/unfair or biased/unbiased
behavior.

## Inputs, provenance and prediction reuse

- Development: frozen processed NHANES 2011–March 2020, N=25,580, 3,690 positives.
- Temporal: previously evaluated frozen NHANES 2021–2023, N=7,536, 1,070 positives.
- Development probabilities are reused from
  `results/nhanes/threshold_oof_predictions.csv`, filtered to XGBoost. No development
  in-sample predictions are substituted and no OOF model is fitted again.
- OOF metadata is checked against the development CSV hash, frozen-model provenance
  hash for the saved prediction file, complete Core-10 specification, median
  preprocessing, no class weighting/calibration and all frozen XGBoost parameters.
- OOF rows are joined by `seqn`, with one-to-one coverage, unique IDs, matching
  targets/cycles and reconstructed deterministic fold membership from the saved
  5-fold stratified design (shuffle=True, seed 42). Reconstructing fold indices is
  an integrity check only, not training.
- Temporal probabilities use `load_frozen_model` and the existing
  `predict_probabilities(..., pipeline=...)` helper. No SHAP/recommendation layer or
  second predictive implementation is needed. The temporal CSV hash must match
  its previous evaluation provenance. Participant overlap between cohorts fails.

The saved OOF probabilities come from five training-fold pipelines with the fixed
XGBoost configuration. They are not predictions from the single final full-development
artifact on its own training participants. Because the configuration was selected
in earlier development work, this is also not a newly unbiased, fully nested
subgroup model-selection estimate. The bootstrap conditions on those saved fits.

The post-hoc temporal overall ROC-AUC, average precision and Brier score reproduce
the previously recorded aggregate results exactly (differences 0.0). This is an
additional inference/provenance consistency check, not new evidence that the
temporal cohort was unseen.

## Pre-specified subgroup definitions

| Family | Definition | Interpretation |
|---|---|---|
| Sex | Existing NHANES sex codes 1=Male, 2=Female | Demographic grouping, not comprehensive gender identity |
| Age | [20,40), [40,60), >=60; displayed 20–39, 40–59, 60+ | Demographic grouping |
| Race/ethnicity | Existing RIDRETH3-derived categories below | Demographic/social categories |
| BMI | <25, [25,30), >=30; missing BMI retained separately | Clinical subgroup, not a protected-attribute fairness category |
| Completeness | All ten original predictors observed versus >=1 missing | Robustness grouping, not a fairness category |

The race variable is `RIDRETH3`, renamed by `src/data/load_nhanes.py`. Valid codes
are checked by the existing preprocessing, and labels are reused directly from
`src/decision_support/prototype_presentation.py:CATEGORY_LABELS`:

| Code | Existing project label |
|---:|---|
| 1 | Mexican American |
| 2 | Other Hispanic |
| 3 | Non-Hispanic White |
| 4 | Non-Hispanic Black |
| 6 | Non-Hispanic Asian |
| 7 | Other race / multiracial |

No new race mapping is invented. Unexpected nonmissing codes fail clearly. Expected
categories remain in the output even if empty or single-class; missing categories
are retained separately when present. Each family must partition the entire
cohort exactly once. Across families, participants occur repeatedly, so those
rows are not independent samples.

BMI and completeness are based on original observed values **before** frozen
imputation. BMI <25 includes all observed values below 25, not a diagnosis of
healthy weight. Missing BMI is not silently excluded or assigned to an imputed
category. There are 17 output rows per cohort in this audit, including missing BMI.

## Fixed threshold and metrics

All primary thresholded metrics use the exact value retrieved from frozen metadata:

`sensitivity_80_reference = 0.14026810228824615`

Prediction is positive when probability >= that value, including exact ties. This
reference existed before this subgroup analysis; there is no subgroup-specific
selection, optimization or default-0.5 secondary table. Its name describes the
overall development operating reference, not guaranteed sensitivity in every
subgroup or in a new cohort.

Every row includes N, positive/negative counts, prevalence, ROC-AUC, average
precision, Brier score, sensitivity, specificity, PPV (`precision`), NPV (`npv`),
F1, TP/FP/TN/FN, threshold, subgroup type, prediction source and instability flags.
Average precision is the stepwise PR summary, not trapezoidal PR area. Undefined
ratios are missing, not forced to zero. AUC/AP are missing for single-class groups.

## Bootstrap 95% confidence intervals

Settings were specified before inspecting subgroup performance:

- Base random seed **42**; **1,000 ordinary participant-bootstrap replicates** per
  eligible subgroup, with replacement and no outcome stratification.
- Separate deterministic subgroup seeds are the first four little-endian bytes of
  SHA256(`base_seed|cohort|family|group_code`). They are saved in each row and do not
  depend on observed performance or execution order.
- Existing (target, probability) rows are resampled together. Multiplicity counts
  of sampled participant indices represent the ordinary replacement bootstrap
  exactly; the implementation sorts scores once and calculates tie-aware weighted
  AUC/AP. Tests compare it against explicit resampled sklearn metrics, including ties.
- Report percentile 2.5% / 97.5% intervals using linear quantile interpolation for
  AUC, AP, Brier, sensitivity and specificity. No predictor/preprocessor is refitted.
- One-class samples are excluded for AUC/AP; sensitivity/specificity use their own
  valid class denominators and Brier remains defined. Valid-draw counts and one-class
  draw counts are recorded. A metric needs at least 95% valid replicates for its CI.
- Pre-specified conservative count policy: flag a group if N<100, positives<30 or
  negatives<30. Keep its row and valid point estimates, suppress **all** its CIs,
  and record the count reason. These are analysis heuristics, not clinical cutoffs
  or a guarantee of statistical stability above the limits.

No observed group met that suppression policy in either cohort. Nonetheless,
temporal ages 20–39 have only **48 positives**, and Non-Hispanic Asian participants
have **49 positives**. Their intervals are relatively wide and merit caution.
The youngest group's sensitivity interval is approximately 0.040–0.234. Absence of
an instability flag is not evidence of precise estimates or acceptable performance.

These are unweighted analytic-cohort results. IID participant bootstrap does not
account for NHANES survey weights, PSU/household clustering or stratification, and
does not produce design-based national-population estimates. It also omits model
fitting, prior tuning and CV-fold uncertainty. Intervals are pointwise, without
simultaneous coverage or multiple-comparison correction. No inferential test of
group differences is performed; CI overlap alone is not such a test.

## Development findings: saved OOF predictions

- Sex AUC: Male 0.8316, Female 0.8202. Sensitivity is 0.8333 versus 0.7636, with
  specificity 0.6648 versus 0.7184.
- Age sensitivity varies from **0.1095** (20–39; 210 positives) to **0.9323** (60+).
  Specificity moves from 0.9895 to 0.2999. Within-age AUC is 0.7387–0.7530; restriction
  of an important predictor's range changes the discrimination task relative to
  pooled AUC.
- Race/ethnicity AUC is 0.8110–0.8531; sensitivity is 0.6976–0.8413. These are
  descriptive differences conditional on the study outcome and case mix.
- BMI observed <25 has AUC 0.8551; missing BMI has 0.7737. At the same reference,
  sensitivity is 0.5910 for BMI <25 and 0.8705 for BMI >=30.
- Complete Core-10 inputs: N=20,588, AUC **0.8320** (95% CI 0.8247–0.8396),
  Brier 0.09543. At least one missing input: N=4,992, AUC **0.8026** (0.7861–0.8184),
  Brier 0.11290.

## Temporal findings: POST-HOC / EXPLORATORY

| Subgroup | N | Positives | AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Male | 3,365 | 517 | 0.8006 | 0.8143 | 0.6204 |
| Female | 4,171 | 553 | 0.8220 | 0.7722 | 0.6987 |
| Age 20–39 | 2,055 | 48 | 0.8137 | 0.1250 | 0.9890 |
| Age 40–59 | 2,152 | 295 | 0.7461 | 0.6000 | 0.7437 |
| Age 60+ | 3,329 | 727 | 0.7513 | 0.9147 | 0.3570 |
| Non-Hispanic Black | 959 | 190 | 0.7809 | 0.8211 | 0.5982 |
| Non-Hispanic Asian | 386 | 49 | 0.8560 | 0.7755 | 0.7448 |
| BMI <25 | 1,561 | 105 | 0.8585 | 0.5619 | 0.8963 |
| BMI 25–<30 | 1,853 | 234 | 0.8155 | 0.7479 | 0.7276 |
| BMI >=30 | 2,338 | 504 | 0.7817 | 0.8552 | 0.4984 |
| Missing BMI | 1,784 | 227 | 0.7550 | 0.8062 | 0.5768 |
| Complete Core-10 | 4,969 | 699 | 0.8314 | 0.7883 | 0.7012 |
| >=1 missing Core-10 | 2,567 | 371 | 0.7712 | 0.8005 | 0.5924 |

This compact discussion table is not the complete subgroup output. All six
race/ethnicity categories, every requested metric, counts and all five CI pairs
are retained in `temporal_subgroup_metrics.csv`; no poorer-performing subgroup
was removed from the audit.

The global reference detects **6 of 48** diagnosed-positive participants aged
20–39 in the temporal cohort (42 false negatives). This low sensitivity is an
important limitation even though young-group AUC is relatively high and Brier is
low. AUC assesses ranking, while a fixed threshold determines classification;
low outcome prevalence also affects loss and predictive values. No threshold
change, clinical assurance or subgroup-specific recommendation follows from this
post-hoc observation.

Temporal race AUC spans 0.7809 (Non-Hispanic Black) to 0.8560 (Non-Hispanic Asian).
Corresponding AUC CIs are approximately 0.7469–0.8145 and 0.8026–0.9010. These
point differences and pointwise intervals do not establish a causal race effect
or certify a fairness conclusion.

Completeness remains an important robustness contrast: temporal complete-input
AUC is **0.8314** (0.8157–0.8468) versus **0.7712** (0.7504–0.7925) with missing
inputs; Brier scores are 0.09584 versus 0.10809. The missing-input group has slightly
higher sensitivity but materially lower specificity, so missingness is not a
uniform improvement or decline on every metric. No causal missingness effect is
estimated.

## Descriptive performance ranges

These are max-minus-min point estimates across every finite subgroup estimate,
with no inference or certification attached:

| Family | Development AUC range | Temporal AUC range | Development sensitivity range | Temporal sensitivity range |
|---|---:|---:|---:|---:|
| Sex | 0.0114 | 0.0214 | 0.0697 | 0.0422 |
| Age | 0.0142 | 0.0675 | 0.8228 | 0.7897 |
| Race/ethnicity | 0.0421 | 0.0752 | 0.1437 | 0.0458 |
| BMI, including missing | 0.0814 | 0.1036 | 0.2795 | 0.2933 |
| Completeness | 0.0294 | 0.0602 | 0.0268 | 0.0123 |

Largest specificity variation is by age: 0.6896 development and 0.6320 temporal.
Largest Brier variation is also by age: 0.1462 and 0.1245. Among observed BMI bins
alone, temporal AUC range is about 0.0768; the larger 0.1036 includes missing BMI
because that group must not be silently discarded. The range files also report
max/min group labels, instability flags and stable-only ranges. A range is missing
if fewer than two finite group estimates exist, rather than misleadingly zero.

AP, PPV, NPV and Brier depend on prevalence/case mix. Thus their subgroup differences
should not automatically be read as model calibration differences or discrimination
by the system. Race/ethnicity, sex, age and BMI are already predictive inputs;
conditioning on them changes the score/case-mix distribution. In particular,
race/ethnicity is a social/survey category in the frozen model, not a biological
causal interpretation. Diagnosed-status ascertainment may also reflect access to
care and detection differences. This audit cannot isolate those mechanisms.

## Outputs and rerun commands

```powershell
.venv\Scripts\python.exe scripts/run_nhanes_subgroup_audit.py --bootstrap-replicates 1000 --seed 42
.venv\Scripts\python.exe -m pytest tests/test_nhanes_subgroup_audit.py -q -p no:cacheprovider --basetemp=.test_tmp_subgroup_unit
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.test_tmp_subgroup_full --junitxml=results/nhanes/subgroup_audit/pytest_full_report.xml
```

For the complete suite in the restricted Windows sandbox, local socket permission
is required by the existing Streamlit tests. Unlike earlier tasks, the read-only
raw-cohort integration module is included here: this audit explicitly permits
post-hoc temporal data access. It does not write XPTs or processed cohorts.

New implementation files:

- `src/evaluation/nhanes_subgroup_audit.py`
- `scripts/run_nhanes_subgroup_audit.py`
- `tests/test_nhanes_subgroup_audit.py`
- This methodology document.

Analysis artifacts under `results/nhanes/subgroup_audit/`:

- `development_subgroup_metrics.csv`, `temporal_subgroup_metrics.csv`
- `development_disparity_summary.csv`, `temporal_disparity_summary.csv`
- `subgroup_audit_summary.json`, with fixed settings, seed derivation, cohort and
  artifact hashes before/after, OOF/temporal source paths, package/source versions,
  aggregate reproducibility deltas, warnings, instability policy and limitations.
- `figures/subgroup_roc_auc`, `subgroup_sensitivity`, `subgroup_specificity`, and
  `completeness_comparison`, each as 300-dpi PNG and vector PDF, produced using
  matplotlib only. Cohorts are labelled separately; temporal panels explicitly
  say POST-HOC / EXPLORATORY. Stars/X markers would identify count-flagged groups
  without suppressing their point estimates.

Tests verify boundaries, missingness partitioning, existing category mappings,
tie-aware bootstrap agreement with explicit sklearn resampling, deterministic
percentile intervals, one-class draw handling, suppression of unstable CIs,
fixed threshold counts, ID-aligned OOF reuse, temporal-only frozen inference,
output schemas, descriptive ranges and all figure exports. Frozen estimator and
preprocessing fit methods are blocked during the new tests. Frozen artifact and
metadata hashes are checked before and after.

Verification results: **10 new tests passed** (final targeted run: 20.45 seconds).
The complete suite, including the real read-only raw-cohort integration module,
passed **168 tests in 82.45 seconds**, with zero failures/errors/skips. A subsequent
provenance-only guard was covered by the final targeted run: the runner hashes
only analysis files it owns, so independently regenerated pytest reports cannot
make analysis-output hashes stale. The final four numerical CSV tables were
verified byte-identical after the figure/provenance cleanup. The four figure types
were exported to PNG/PDF and the sensitivity/completeness layouts visually checked.
No runtime warnings were recorded in the saved audit.

The audit's previous-results hash guard protects the existing baseline, tuning,
temporal, XAI and CDSS reports from being overwritten. The artifact remains
`fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`.
Metadata remains
`57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3`.
All source/output hashes in the final analysis summary were verified against the
saved files. No frozen artifact, cohort, metadata, reference, model code or prior
result was modified. The Streamlit app and recommendation framework are unchanged.

## Limitations and conclusion

The findings are a conditional, unweighted, retrospective description of the
frozen model's errors and probability losses. They do not establish fairness,
causality, clinical utility, prospective generalization or suitable new operating
points. The performance variation—especially low young-adult sensitivity at the
global reference and weaker discrimination with missing inputs—should remain
visible in study reporting. No subgroup is excluded, no subgroup threshold is
optimized, and no model or clinical recommendation is changed in response.

# Complementary local LIME explanations for the frozen NHANES model

## Scope and use

LIME is included to complement the existing exact Tree SHAP explanations of
`nhanes_phase2_xgb_v1`. This is an explanation exercise, not model development.
The predictor, preprocessing, calibration, and operating references are unchanged.
Neither the reusable module nor the runner accepts a temporal-cohort path.
Streamlit is unchanged and continues to use the existing SHAP backend.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-nhanes-lime.txt
.venv\Scripts\python.exe scripts/run_nhanes_lime_local.py
```

LIME version: **0.2.0.1**. Existing numerical/XAI package versions are preserved
by the inherited constraints. For an arbitrary single patient, reuse an instance:

```python
from scripts.demo_nhanes_cdss import synthetic_profiles
from src.explainability.nhanes_lime_local import NhanesLimeLocal, compare_shap_lime

explainer = NhanesLimeLocal()
patient = synthetic_profiles()["intermediate_scoring_example"]
result = explainer.explain_patient(patient, random_seed=42, num_samples=5000)
comparison = compare_shap_lime(result["shap_result"], result)
```

The same method accepts a complete Core-10 mapping or a one-row DataFrame. It
reuses the backend's validation and supports missing predictor values. All fields
must still exist. The reference cohort is loaded once per service instance;
the explainer can be reused for sequential patient requests.

## Reference data and original feature space

Use the entire frozen development cohort (NHANES 2011–March 2020, N=25,580),
without subsampling, as LIME's reference distribution. The existing development
loader validates cohort integrity. Targets and identifiers are not passed to LIME.
The temporal cohort is not read, inspected, or used for reference construction,
parameter choice, or explanation evaluation.

The already-fitted numeric median and categorical most-frequent imputers
**transform only**. This gives ten original features with finite reference values,
before one-hot encoding. Neither imputer is fitted again. Missing patient values
are similarly transformed using those frozen imputers; their explanation concerns
the imputed input state, not an observed measurement. Each output retains the
original value, missingness flag, and reference-imputed value.

The fitted encoder's category lists define the allowed codes:

| Feature | NHANES codes | Temporary LIME indices |
|---|---|---|
| sex | 1, 2 | 0, 1 |
| race_ethnicity | 1, 2, 3, 4, 6, 7 | 0, 1, 2, 3, 4, 5 |
| smoking_status | 0, 1, 2 | 0, 1, 2 |

These temporary indices avoid treating noncontiguous category codes as numeric
distances. LIME is explicitly told which original columns are categorical.
Every perturbation is decoded back to the exact NHANES codes, then passed to the
existing `predict_probabilities(..., pipeline=backend.pipeline)` helper. The frozen
pipeline performs its normal imputation/encoding/XGBoost inference. No second
predictive implementation is introduced. The callback returns the class-0 and
class-1 probabilities in that order.

LIME terms map back by integer feature ID, not by parsing rounded interval text.
All ten original features are returned; the 18 one-hot columns are not exposed.
For example `race_ethnicity=3` is a category condition, not a numeric trend.

## Fixed LIME method

- Primary seed: **42**; **5,000 perturbed samples per explanation**, including
  the original input as LIME's first sample.
- Classification mode; explain class 1, physician-diagnosed diabetes.
- Numeric variables use development-derived quartile discretization.
- Categorical values and numeric bins are sampled independently using empirical
  reference frequencies. Numeric samples are inverse-discretized within bins
  using LIME's reference-based truncated-normal mechanism.
- Interpretable features indicate whether a sample matches the patient's numeric
  bin or categorical value. All ten indicators are one for the explained patient.
- Euclidean distance in this interpretable space; kernel width
  `0.75 * sqrt(10)`; LIME's default kernel
  `sqrt(exp(-distance**2 / kernel_width**2))`.
- `feature_selection="none"`, ten returned features. The default distance-weighted
  Ridge surrogate uses alpha=1 and an intercept. Its target is the frozen model's
  probability, never the observed diabetes outcome.
- A fresh seeded LIME object is constructed for each request, so RNG state from
  previous patients does not affect reproducibility. Ties use fixed Core-10 order.

LIME fits its own reference statistics/scaler and weighted local surrogate; those
are explanation machinery, not refitting the frozen model or preprocessing.
There is no search over sample count, kernel width, bins, or model parameters.
The settings were fixed before examining these synthetic results and were not
altered to improve agreement or local fit.

## Output and interpretation

`explain_patient` returns the unchanged frozen probability, model version, seed,
sample count, local intercept, ten signed local weights/conditions, absolute
importance, rank, and direction. Positive coefficients are labelled
`increases_model_probability`, negative coefficients
`decreases_model_probability`, and absolute values <=1e-10 are neutral. These
names refer to the local surrogate on a probability-output scale. They do not
mean a causal effect, an intervention effect, or a guaranteed change in the
frozen model's probability.

`surrogate_prediction` is the **unbounded local surrogate output**, not another
valid diabetes probability. Because all active indicators are one, it equals
intercept plus the sum of ten weights; this identity is checked. It is not
required to equal the frozen probability and is deliberately not clipped to
[0,1]. `surrogate_absolute_error` records the discrepancy at the patient, while
`local_weighted_r2` describes fit over the sampled, weighted neighborhood. The
latter is descriptive, in-sample surrogate fidelity, not an independent
validation score or explanation confidence.

The callback's probability for the original patient is checked against the
existing backend probability. The returned `shap_result` is that backend's full
existing explanation, including its raw-margin and probability integrity checks.
SHAP remains primary because it explains this frozen tree ensemble exactly on
its raw-margin scale with verified additivity. LIME is an approximate probability
surrogate and, in this run, has important fidelity limitations.

## SHAP–LIME comparison and stability

Both explanations are ranked by absolute magnitude at original Core-10 level.
SHAP first sums signed one-hot contributions per feature, as before. For k=3 or
5, overlap is intersection size divided by k. Direction agreement is the fraction
of features in that intersection with matching signs (same tiny neutral tolerance
for both methods); an empty intersection gives null, not zero. The CSV reports
top-3 and top-5 intersection agreement separately, plus agreement over all ten
features. Magnitudes are never directly equated: SHAP uses log-odds, whereas LIME
weights are coefficients in a probability-output surrogate with another baseline.

The modest stability check repeats each profile at seeds 42, 43, 44, 45, 46 with
all other settings fixed. It records per-feature top-3/top-5 selection counts and
frequencies, plus mean pairwise set overlaps (ten seed pairs per profile). This
checks sampling stability for three synthetic profiles only. It does not establish
clinical validity, fidelity, rank-order stability, or population-wide robustness.

## Recorded synthetic results

These are the same fabricated software examples as the existing backend, not
clinical patient archetypes. Feature lists below are ordered by absolute weight.

| Example | Frozen probability | LIME top 3 | SHAP top 3 | Top-3 overlap | Top-5 overlap | Direction agreement in top-3 / top-5 intersections |
|---|---:|---|---|---:|---:|---:|
| Lower scoring | 0.5871% | age, waist_cm, race_ethnicity | age, waist_cm, hdl_mg_dl | 2/3 | 4/5 | 100% / 100% |
| Intermediate scoring | 6.5429% | age, race_ethnicity, mean_diastolic_bp | race_ethnicity, waist_cm, mean_diastolic_bp | 2/3 | 4/5 | 100% / 100% |
| Higher scoring | 52.5265% | age, total_cholesterol_mg_dl, waist_cm | age, total_cholesterol_mg_dl, waist_cm | 3/3 | 5/5 | 100% / 100% |

All-feature sign agreement was 100%, 90%, and 100%, respectively. Both top-3 and
top-5 sets were identical across the five seeds for each profile: mean pairwise
overlap 1.0. Every primary top-3 member was selected 5/5 times; complete top-5
frequencies are in the stability CSV. This is set stability, not equality of
coefficients or order across seeds.

| Example | Local intercept | Unbounded surrogate prediction | Frozen probability | Absolute discrepancy | Weighted neighborhood R² |
|---|---:|---:|---:|---:|---:|
| Lower scoring | 0.245223 | -0.156041 | 0.005871 | 0.161912 | 0.392431 |
| Intermediate scoring | 0.198675 | 0.002790 | 0.065429 | 0.062639 | 0.111362 |
| Higher scoring | 0.056565 | 0.427685 | 0.525265 | 0.097581 | 0.462976 |

The surrogate's approximately 6–16 percentage-point discrepancies and the
negative lower-profile output are substantial limitations. Stable feature
selection and overlap must not be interpreted as accurate reproduction of the
model at that patient. No explanation setting was subsequently tuned to conceal
these discrepancies. There were no runtime warnings in the saved primary run.

## Saved artifacts, verification, and clinical limitations

- `results/nhanes/xai/lime_synthetic_explanations.csv`: 30 primary-seed rows,
  one per profile/feature, with conditions, weights, ranks and fidelity diagnostics.
- `results/nhanes/xai/shap_lime_agreement.csv`: three profile-level comparisons.
- `results/nhanes/xai/lime_stability.csv`: 30 profile/feature rows of frequencies;
  pairwise summary values repeat for each feature of a profile.
- `results/nhanes/xai/lime_provenance.json`: immutable input hashes before/after,
  prior XAI result hashes, reference size/codes, exact settings, seeds, synthetic
  inputs, per-seed diagnostics, package/source versions, timestamps and warnings.

Tests prohibit fitting the frozen model and its preprocessing, permit only the
fixed development CSV read, verify callback delegation and category decoding,
unchanged probabilities/hashes, deterministic rankings, missing-value support,
overlap/sign calculations, stability calculations, and output/provenance schemas.
The full regression run excludes `tests/test_nhanes.py`, whose raw integration
checks would reopen temporal data. The existing Streamlit smoke tests require
local socket permission when run within the restricted Windows sandbox.

Verification result: **78 tests passed in 106.41 seconds**, including six new
LIME tests. `pip check` reported no broken requirements. Model, metadata, and
development-cohort SHA-256 hashes matched before/after execution; the artifact
remains `fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`.
No frozen-model or frozen-preprocessing fitting occurred. Only the development
cohort was used; no temporal data were accessed. The Streamlit source and
existing SHAP backend were not modified.

SHAP and LIME need not agree: they use different explained outputs, baselines,
neighborhoods, and additive representations. Disagreement alone does not prove
either method wrong; LIME's approximation error must also be examined. Independent
perturbations can violate feature correlations and create implausible profiles;
quartiles can obscure within-bin behavior. Development-derived frequencies are
not survey-weighted population estimates. Reference imputation and category coding
affect explanations. Three synthetic examples are insufficient for broader claims.
Neither method establishes causation, clinical utility, diagnosis, or a validated
operating threshold. No clinical recommendations or risk categories are created.

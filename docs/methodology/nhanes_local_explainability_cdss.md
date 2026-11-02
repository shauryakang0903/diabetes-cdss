# Frozen-model patient-level decision-support backend

Current extension: the backend now appends model-derived strata and a separate
guideline-informed recommendation layer. See
`docs/methodology/nhanes_risk_recommendation_framework.md`. The original prediction,
SHAP workflow and integrity checks described below remain unchanged.

`src.decision_support.nhanes_diabetes_cdss.NhanesDiabetesCDSS` is the prototype
backend. Construct one instance, then call `predict(patient)` for each sequential
request. It loads and verifies the existing versioned artifact and metadata via
the frozen loader, and caches its fitted pipeline, feature mapping, and exact
TreeExplainer. It never reads a cohort, fits preprocessing, retrains, recalibrates,
or changes model parameters. Keep the recorded artifact environment and install
the existing `requirements-nhanes-xai.txt` dependencies. For concurrent serving,
use a separate service instance per worker; shared threaded use is not promised.

```python
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS

service = NhanesDiabetesCDSS()
result = service.predict({
    "age": 48, "sex": 1, "race_ethnicity": 3, "bmi": 28,
    "waist_cm": 98, "mean_systolic_bp": 128, "mean_diastolic_bp": 80,
    "hdl_mg_dl": 48, "total_cholesterol_mg_dl": 185, "smoking_status": 1,
})
```

## Input contract

Accept a mapping of scalar values or a one-row pandas DataFrame containing all
Core-10 fields. Missing values (`None`, NaN) are allowed, including all predictors
missing; they pass through the frozen imputers. This is software compatibility,
not evidence that an all-missing prediction is clinically useful. Missing fields,
duplicate columns, infinities, nonnumeric values, boolean codes, multiple patients,
and nonnumeric categories raise errors. Numeric strings are accepted by the
existing frozen input validator. No new clinical range exclusions are imposed.
Extra fields are ignored by preprocessing and never echoed in output.

Harmonized NHANES codes (not text labels): sex 1 male / 2 female;
race_ethnicity 1 Mexican American / 2 other Hispanic / 3 non-Hispanic White /
4 non-Hispanic Black / 6 non-Hispanic Asian / 7 other race including multiracial;
smoking_status 0 never / 1 former / 2 current. Allowed nonmissing category values
are read from the fitted encoder. The post-hoc input-quality extension now permits
unknown numeric codes when the frozen encoder can process them, and flags limited
applicability; the frozen encoder still has `handle_unknown="ignore"`. See
[input-quality layer](../validation/nhanes_input_quality_layer.md) for the policy change.
Units are years, kg/m2, cm, mmHg, and mg/dL as indicated by the feature names.

## Prediction and operating references

The output dictionary is JSON-safe (`json.dumps(result, allow_nan=False)`). It
contains `model_version`, `predicted_probability`, percentage, and a technical
0.5 `default_classification`. This estimates the frozen DIQ010 physician-diagnosed
diabetes target, not future disease incidence or a clinical diagnosis.

`thresholds` contains the exact metadata references and an `at_or_above` flag
for each (equality meets the reference). The sensitivity-90 reference
0.08646801114082336 is lower than sensitivity-80 reference
0.14026810228824615; Youden reference is 0.12872706353664398. These numbers are
loaded, never reselected. `reference_interpretation` describes relationships to
the sensitivity-oriented references. Flags also expose the Youden relationship.
These are development-derived operating references, not clinically validated
cutoffs or a recommended production operating point. The later Objective-3 layer
reuses the exact sensitivity-oriented values to label LOW/MODERATE/HIGH
**model-derived** strata; their clinical validity has not been established.

## Local explanation and integrity

Use the same exact tree-path-dependent Tree SHAP as global analysis, with the
frozen booster and no external background data. Explain raw margin/log-odds for
class 1. Contributions are not probability changes and do not imply causation.
The shared mapping uses fitted transformer slices and encoder category ordering;
all 18 transformed contributions are summed with their signs into the original
10 features before taking absolute values. This preserves cancellation across
one-hot columns and preserves each patient's contribution total.

`all_feature_contributions` has feature name, original value (missing is JSON
null), missingness flag, signed contribution, absolute magnitude, and direction.
It is sorted by decreasing absolute magnitude, with Core-10 input order resolving
ties. Only absolute values at most 1e-10 are neutral. Increasing/decreasing refers
to model score relative to the SHAP base, never causal risk/protection. The top
three entries of each sign are returned separately; there may be fewer than three.

Shared global additivity verification checks base plus transformed SHAP against
the booster margin (atol 1e-4, rtol 1e-5). Signed Core-10 and transformed totals
must agree (atol/rtol 1e-10). Independently apply the logistic transformation to
base plus Core-10 sum and check against pipeline `predict_proba` (atol 1e-6,
rtol 1e-5). `explanation_integrity` returns margins, totals, reconstruction errors,
dimensions, scale, and tolerances. Failures raise rather than returning an
unchecked explanation. Floating-point residuals are expected.

## Demonstration and validation

Run `.venv/Scripts/python.exe scripts/demo_nhanes_cdss.py`. Three fabricated
lower-, intermediate-, and higher-scoring examples print complete calculated
results. Their labels are software demonstration descriptions, not validated
patient archetypes. No expected probabilities are hard-coded or real identifiers
used. The service does not persist patient records or results.

Calculated software examples (the script always recalculates these):

| Synthetic example | Probability | Sensitivity-reference relationship | Top increasing features | Top decreasing features |
|---|---:|---|---|---|
| Relatively lower scoring | 0.5871% | Below both | sex (only one increasing) | age, waist_cm, hdl_mg_dl |
| Intermediate scoring | 6.5429% | Below both | hdl_mg_dl, smoking_status, age | race_ethnicity, waist_cm, mean_diastolic_bp |
| Higher scoring | 52.5265% | At or above both | age, total_cholesterol_mg_dl, waist_cm | race_ethnicity, mean_diastolic_bp, smoking_status |

Across these three examples the largest raw-margin reconstruction error was
1.4702090993523598e-6 and probability reconstruction error was
1.0961807428433445e-7. Directions are model associations; for example, a lower
cholesterol contribution must not be translated into causal treatment advice.

Tests cover fit prohibition, cohort-read prohibition, immutable model/metadata
hashes, cached loading, input errors/missingness, probability bounds, ten-feature
aggregation, contribution totals/directions/order, both integrity checks, exact
metadata references including equality, deterministic results, and JSON output.
The existing raw-cohort integration module `tests/test_nhanes.py` is excluded from
the regression run to avoid its raw temporal-cohort loading. No temporal dataset
is needed for backend validation.

## Limitations

Every result explicitly states: this is model-based support, not diagnosis;
development-derived references are not clinically validated cutoffs; SHAP
explains model behavior rather than causation; incomplete inputs may reduce
robustness, reflecting the previously completed temporal-validation finding
of lower performance with missing Core-10 predictors. That finding is carried
forward as a limitation without reopening temporal data here. Intended use is
adult research support. A separate rule layer now supplies guideline-informed
educational follow-up/lifestyle prompts, with no treatment prescribing or new
production threshold. This backend does not establish clinical utility, deployment
readiness, or prospective risk validity.

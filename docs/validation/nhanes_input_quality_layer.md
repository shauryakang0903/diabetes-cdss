# Input completeness and development applicability

This informational layer accompanies the frozen NHANES Phase-2 XGBoost model. It does not change prediction, preprocessing, model parameters, reference thresholds, SHAP, risk strata or recommendation rules. It is not formal uncertainty quantification or a probability that a prediction is correct.

The exact Core-10 inputs remain age, sex, race_ethnicity, bmi, waist_cm, mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl, total_cholesterol_mg_dl and smoking_status. All ten keys are required by the backend; missing values are permitted and normalized to NaN for the existing stored median/mode imputers. Nothing is fitted.

## Development-only reference

`python -m scripts.generate_nhanes_input_reference_metadata` creates `models/nhanes/phase2_input_reference_metadata.json` from only the frozen development CSV (25,580 adults). No temporal observations are read. Generation verifies its frozen checksum and outcome counts. Statistics use observed raw values before imputation: count, median, p01, p05, p95, p99, minimum and maximum; NumPy's linear quantiles are deterministic. Categorical sets contain observed development codes. The JSON records the source path/hash, frozen model and metadata hashes, feature names, method/version and label definitions. Existing differing reference contents fail rather than being silently replaced. Normal inference reads the saved reference JSON, never a cohort.

Numeric values within inclusive p01–p99 are `typical`; values outside that interval but within inclusive observed min/max are `uncommon`; values beyond observed min/max are `out_of_development_range`. Missing inputs are `missing`. Categorical inputs are `known_category`, `unseen_development_category` or `missing`. These checks do not clip, replace or normalize supplied values.

## Interface categories

Completeness reports counts and percentage: 10 observed = **Complete**; 8–9 = **Minor missingness**; 0–7 = **Reduced completeness**.

Overall labels are project-defined software/interface labels:

- **Within development-data profile**: complete inputs, typical numeric values and known categories.
- **Interpret with caution**: minor missingness or uncommon values without stronger flags.
- **Limited model applicability**: reduced completeness, any out-of-range value or any unseen category; this takes precedence.

These are not clinical validity categories. The visible disclaimer is: “This indicator describes similarity to the model's development data. It is not a clinical confidence score and does not quantify the probability that the prediction is correct.”

## Backend and interface

The backend appends `input_quality` after calculating existing prediction, SHAP and recommendations. Regression fixtures captured before this change cover complete and incomplete synthetic profiles; all earlier result fields must remain exactly equal. Model artifact and metadata checksums are unchanged.

The previous service rejected unseen numeric category codes before preprocessing. This validation policy intentionally changes: the frozen encoder uses `handle_unknown="ignore"`, so a processable unseen numeric code now produces a visible applicability flag and the encoder's existing all-zero category representation. No new category is learned. Non-numeric, boolean, infinite, non-scalar inputs and missing keys still fail. Unknown categories are not silently recoded to known groups. The separate LIME categorical perturbation interface retains its existing restrictions.

Streamlit keeps age, sex and race/ethnicity required. Its “Unavailable clinical measurements” selector can mark BMI, waist, blood pressures, lipids or smoking status missing; disabled controls retain their previous values but submit None. Set availability before editing the form. Changing availability clears stale output. Predictions use only frozen imputation. The result shows “Input Data Quality”, completeness, overall label, flagged fields, the disclaimer and an expandable ten-feature detail section. Existing risk, SHAP and recommendation displays remain present. The UI offers only known categorical options; unseen numeric categories are a backend-supported case.

## Motivation and limits

The previous post-hoc subgroup audit reported temporal ROC-AUC approximately 0.8314 with complete Core-10 inputs and 0.7712 with at least one missing input. These findings motivate explicit completeness reporting; the groups may differ in other characteristics, so this does not establish that missingness caused the difference. Temporal results were already known before this layer was created: this is a post-hoc engineering enhancement, not independent temporal validation. No temporal cohort is used to derive ranges or test this layer.

Development distributions are population-specific. Percentile boundaries and completeness cutoffs are descriptive choices, not validated decision thresholds. Marginal checks do not assess joint distribution shift, measurement correctness, causal relevance, demographic fairness or individual error probabilities. NHANES age top-coding limits interpretation of the observed age maximum. In-range inputs do not guarantee correct predictions; out-of-range inputs do not guarantee incorrect predictions. Missing-value imputation preserves operation, not necessarily performance. This layer has no prospective clinical validation.

## Verification

Focused tests exercise completeness, exact percentile and min/max boundaries, categories, deterministic generation, provenance failures, JSON schema, cohort-free inference and strict preservation of pre-layer result fields. Streamlit AppTest covers complete low/high examples and missing HDL with uncommon BMI; a separate temporary server checks the HTTP health endpoint. Full-suite and server evidence are saved separately under `results/nhanes/input_quality/`, without regenerating previous analyses.

Launch from the project root with `.venv\Scripts\python.exe -m streamlit run app.py`.

The completed verification run passed 23 focused tests and 207 full-suite tests
(119.18 seconds). The temporary server returned HTTP 200 / `ok` and was stopped.
Evidence is in `results/nhanes/input_quality/pytest_focused_report.xml`,
`pytest_full_report.xml`, `streamlit_health.json` and `integrity_and_examples.json`.
The existing complete lower/higher examples retained probabilities
0.005870712921023369 / 0.5252653956413269. Missing HDL plus BMI 58 retained
0.10987188667058945, with 9/10 completeness and “Interpret with caution”.
All legacy output fields matched the captured pre-layer fixtures exactly.

Frozen artifact SHA-256 before and after:
`fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`.
Frozen metadata SHA-256 before and after:
`57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3`.

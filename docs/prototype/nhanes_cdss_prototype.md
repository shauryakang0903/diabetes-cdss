# Explainable diabetes CDSS prototype

## Purpose

This single-page academic demonstrator connects the completed NHANES research
workflow to a usable patient-level interface. It shows frozen-model classification
probabilities, existing development-derived operating references, and local SHAP
explanations. The Objective-3 extension adds LOW/MODERATE/HIGH **model-derived**
strata using those same references, plus guideline-informed follow-up, monitoring,
lifestyle and measurement-review prompts. It adds no model development, new
probability thresholds, diagnosis or medication prescribing. See
`docs/methodology/nhanes_risk_recommendation_framework.md` for exact rules and sources.

Report description: The prototype integrates a temporally evaluated NHANES-based
XGBoost classifier with patient-level SHAP explanations in a Streamlit interface.
Users enter ten health variables and receive a model-estimated diabetes-classification
probability, model-derived strata, signed feature contributions, and auditable
guideline-informed actions, with visible limitations separating decision support
from diagnosis or treatment prescribing.

## Installation and launch

From the repository root in PowerShell, using the existing project environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-prototype.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
Stop the server with Ctrl+C. Streamlit is pinned to **1.66.0**. The requirements
reuse the existing XAI requirements and constraints to preserve NumPy, pandas,
scikit-learn, XGBoost, SciPy, and joblib versions. Streamlit requires Pillow <13,
so Pillow is pinned to 12.3.0 (previous environment: 13.0.0); this is an image
rendering dependency, not a change to the model or numerical environment.
Usage statistics are disabled in `.streamlit/config.toml`.

Keep the trusted existing artifact and metadata in `models/nhanes/`, alongside
the recorded frozen-model environment. This application does not create a model
if those files are absent. Loading or integrity errors are shown as concise UI
messages; detailed exceptions are logged to the server console.

## Architecture

```text
app.py (Streamlit form and output)
  -> prototype_presentation.assess_patient (direct delegation)
  -> NhanesDiabetesCDSS.predict
  -> existing verified frozen model loader / fitted pipeline
  -> existing shared local/global SHAP mapping and integrity checks
  -> reference-based strata + guideline-informed rules (nhanes_recommendations)
```

`prototype_presentation.py` formats existing backend outputs and plots their
already-aggregated values; it contains no prediction, imputation, threshold
selection, or SHAP algorithm. Frozen version: `nhanes_phase2_xgb_v1`.

The model/explainer are lazily cached as a Streamlit resource after the first
assessment. A lock serializes requests to that shared explainer. Patient results
are held in per-session memory, never in the global resource cache. The app does
not write patient records to disk or read raw/processed cohort CSVs. Inputs are
sent to the local Streamlit server, so this is not a privacy-reviewed production
clinical application. Use synthetic records for evaluation demonstrations.

## Inputs and interaction

The form collects the fixed Core-10 features, with readable labels and units:

| Input | Unit / interface |
|---|---|
| age | years |
| sex | Male / Female, mapped to frozen codes 1 / 2 |
| race_ethnicity | NHANES labels mapped to frozen codes 1, 2, 3, 4, 6, 7 |
| bmi | kg/m² |
| waist_cm | cm |
| mean_systolic_bp | mmHg |
| mean_diastolic_bp | mmHg |
| hdl_mg_dl | mg/dL |
| total_cholesterol_mg_dl | mg/dL |
| smoking_status | Never / Former / Current, mapped to 0 / 1 / 2 |

The original interface required all fields. The input-quality extension now allows
unavailable clinical measurements while retaining required demographics; see
[input-quality layer](../validation/nhanes_input_quality_layer.md). The backend supports missing
measurements, as documented in `docs/methodology/nhanes_local_explainability_cdss.md`.
No new clinical range exclusions are applied. Category controls only translate
labels to the existing codes; backend validation remains authoritative.

The existing three synthetic profiles are reused from `scripts/demo_nhanes_cdss.py`.
Selecting one populates the form and clears any previous result. Profile labels
describe software examples, not clinical archetypes or risk bands. Editing
measurements does not assess the patient. Press **Run Diabetes Assessment** to
calculate a new result. Results explicitly refer to the last submitted inputs.

## Available sections and output interpretation

- Header and visible research/diagnosis disclaimer.
- About the system, explaining the physician-diagnosed DIQ010 target.
- Patient input form and synthetic profile selector.
- Submitted probability in percent and technical model version.
- LOW / MODERATE / HIGH model-derived stratum using the exact sensitivity-90
  and sensitivity-80 references, with a mandatory non-diagnostic disclaimer.
- Backend reference interpretation and table of exact stored thresholds,
  including the 0.50 technical reference, sensitivity-oriented references, and
  Youden statistical reference. No reference is reselected by the app.
- Top contributors increasing and decreasing model score, with original values.
- Signed horizontal Core-10 SHAP chart; orange positive, blue negative, gray
  neutral. Its values are model margin/log-odds contributions, not probability
  changes. The full ten-feature table and reconstruction errors are expandable.
- Previously saved global Core-10 SHAP figure with development-cohort context.
  A missing figure produces an informational message, not a fresh computation.
- Model/validation information using previously recorded aggregate facts:
  development N=25,580, temporal evaluation N=7,536, frozen XGBoost temporal
  ROC-AUC approximately 0.812, AP 0.425, Brier 0.100. No cohort is reopened.
- Visible limitations, plus the backend's returned limitations after assessment.
- Four action tabs after SHAP: clinical follow-up, monitoring, lifestyle, and
  patient-specific considerations, with rule IDs/source labels and an audit expander.

The displayed probability is a classification output for physician-diagnosed
diabetes under the study definition, not an individual clinical diagnosis or
prospective incidence estimate. Sensitivity references retain their development
meaning; the 90%-sensitivity reference has a lower numerical threshold than the
80%-sensitivity reference. No clinical threshold validity is implied.

## Validation and limitations

Tests exercise direct backend delegation, probability formatting, original-feature
table and signed chart rendering, readable category labels, three synthetic
profiles, explicit submission, stored reference text, and concise input/loading/
integrity error paths. Fits and cohort CSV reads are forbidden during the
prototype tests, and frozen artifact/metadata hashes must remain unchanged.
Streamlit's AppTest provides a small application smoke test without heavy browser
automation. Use the recorded regression command:

```powershell
.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_prototype
```

The excluded legacy integration module reads raw temporal data; all other tests
run. Screenshots may be added later for the paper/report.

Implementation verification: **72 tests passed in 80.09 seconds**, including
seven new prototype tests. All three synthetic profiles passed the application
smoke test and actual signed-bar chart rendering. The localhost server returned
`ok` from its health endpoint. `pip check` reported no broken requirements.
The frozen artifact SHA-256 remains
`fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`;
model and metadata hashes were also checked before/after prototype tests.
In the restricted Codex Windows sandbox, Streamlit/AppTest require permission
to use local sockets; the successful application and regression runs used that
permission. This does not require a code workaround in a normal local launch.

This is a local academic prototype with no authentication, clinical deployment
controls, or medical-record integration. It does not replace a qualified health
professional. Reference thresholds are not clinically validated diagnostic
cutoffs. SHAP describes associations in model behavior rather than causation.
Temporal performance was weaker with missing Core-10 measurements. Clinicians
may require laboratory tests and additional information. Guideline-informed
educational prompts are now provided by the Objective-3 extension; they are not
an individual treatment plan. No medications are prescribed and the model-derived
strata are not clinically validated risk categories. Exact sources, triggers,
limitations and scenario results are documented in
`docs/methodology/nhanes_risk_recommendation_framework.md`.

# Model-derived risk strata and guideline-informed decision support

## Objective 3 and scope

This implements Objective 3 of the minor-project synopsis: a patient-specific
stratification and recommendation framework. The operational implementation is
complete as an academic demonstrator; clinical effectiveness and the validity of
these strata have not been established. The predictive model remains
`nhanes_phase2_xgb_v1`. No model fitting, tuning, preprocessing changes,
recalibration, threshold optimization, or temporal-data access was performed.

The new rule layer runs after the existing backend has calculated probability
and verified SHAP. It appends fields to the result while preserving every
pre-existing result field. The endpoint is physician-diagnosed diabetes status
from DIQ010. It is not future diabetes incidence; a negative classification can
also include people with undiagnosed disease. The probability and stratum do not
replace clinical screening or laboratory assessment.

## Exact strata and terminology

Load the existing exact values from
`models/nhanes/phase2_xgboost_metadata.json:reference_thresholds`:

| Label returned by backend/UI | Probability interval |
|---|---|
| LOW model-derived risk | p < 0.08646801114082336 |
| MODERATE model-derived risk | 0.08646801114082336 <= p < 0.14026810228824615 |
| HIGH model-derived risk | p >= 0.14026810228824615 |

The lower boundary is `sensitivity_90_reference`; the upper is
`sensitivity_80_reference`. Lowering a threshold makes more cases positive and
raises sensitivity (while generally reducing specificity). Thus the development
reference for approximately 90% sensitivity is numerically lower than the
approximately 80% reference. The names refer to previously measured development
operating points, not guaranteed sensitivity in new patients or within a stratum.
Exact equality at the lower boundary is MODERATE; exact equality at the upper is
HIGH. Youden and default 0.50 remain existing statistical/technical references
and do not define these three strata. No new probability cutoff is estimated.

Every returned stratum includes this statement, also visible in the UI:

> Low, Moderate, and High are model-derived decision-support strata based on
> development operating references. They are not diagnostic categories or
> clinically validated risk thresholds.

These project-defined labels do not have an ADA clinical endorsement. HIGH is
not a diagnosis or an urgent triage classification; LOW does not rule out diabetes
or waive indicated testing. The output always retains the DIQ010 target caveat.

## Architecture and structured output

```text
Core-10 patient inputs
  -> existing verified frozen XGBoost pipeline probability
  -> existing signed Core-10 SHAP explanation / integrity checks
  -> exact reference-based model-derived stratum
  -> rule evaluation against stratum + observed original input values
  -> structured CDSS output and Streamlit presentation
```

Computing SHAP before the rule layer is a software ordering detail: SHAP signs,
ranks and magnitudes never determine recommendations. `NhanesDiabetesCDSS.predict`
calls `add_recommendation_layer`, which preserves existing fields and adds:

- `model_derived_risk`: level, mandatory label, boundary source, exact reference
  values, interval endpoints/inclusivity and disclaimer. `None` at a boundary
  denotes no additional stratum bound within the probability domain [0,1].
- `recommendations`: four always-present lists: `clinical_follow_up`, `monitoring`,
  `lifestyle`, `feature_aware`. A list can be empty when no corresponding
  measurement rule fires.
- `recommendation_provenance`: framework/model versions, review date, exact
  configuration SHA-256, triggered IDs, input policy and clinician-review caveat.

Each recommendation has a stable rule ID, section, machine-readable trigger,
human-readable trigger description, text, rationale, source IDs and source labels.
Developer source URLs stay in config/documentation, not patient-facing output.
Output remains JSON-safe. `assess_patient` in the UI continues to delegate
directly to the backend, preserving the returned object.

Central definitions are in `configs/nhanes_recommendation_rules.json`, framework
version `nhanes_recommendations_v1`, reviewed 2026-10-18. The evaluator uses a small
explicit set of predicates, not arbitrary expressions, learned models, or inferred
clinical diagnoses. Duplicate rule IDs, missing source labels, incompatible
framework/model versions, invalid probability, and reversed boundaries fail
clearly. Model/explainer caching is retained; the UI cache key now includes the
recommendation framework version.

## Sources and conservative adaptation

Sources were checked on 2026-10-18:

1. [ADA Standards of Care in Diabetes—2026, section 2: Diagnosis and Classification](https://diabetesjournals.org/care/article/49/Supplement_1/S27/163926/2-Diagnosis-and-Classification-of-Diabetes),
   DOI 10.2337/dc26-S002: clinician-directed screening and diagnostic assessment.
   [Authoritative indexed full-text record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690183/)
   and the journal's indexed recommendations establish laboratory/clinical criteria;
   the full journal HTML was access-restricted during verification. The framework
   does not encode diagnostic laboratory cutoffs or a testing schedule.
2. [ADA 2026, section 3: Prevention or Delay](https://diabetesjournals.org/care/article/49/Supplement_1/S50/163924/3-Prevention-or-Delay-of-Diabetes-and-Associated),
   DOI 10.2337/dc26-S003: general activity and individualized nutrition support.
   No model stratum is treated as an established prediabetes diagnosis.
3. [ADA 2026, section 5: Facilitating Positive Health Behaviors](https://diabetesjournals.org/care/article/49/Supplement_1/S89/163932),
   DOI 10.2337/dc26-S005: individualized healthy eating, appropriate activity,
   weight discussion, and tobacco-cessation support. Tobacco recommendation 5.40
   includes cessation support; this prototype encodes counseling/referral only,
   never its pharmacological components. These general principles are adapted
   into educational prompts, not an individual diabetes treatment plan.
4. [NIH/NIDDK: Health Tips for Adults](https://www.niddk.nih.gov/health-information/weight-management/healthy-eating-physical-activity-for-life/health-tips-for-adults):
   conventional adult BMI and waist review conventions.
5. [2025 AHA/ACC multisociety BP guideline summary](https://professional.heart.org/en/science-news/2025-high-blood-pressure-guideline/top-things-to-know):
   conventional elevated BP measurement ranges. No BP diagnosis or drug decision
   is inferred from the entered mean measurements.
6. [NIH/MedlinePlus: Cholesterol Levels—What You Need to Know](https://medlineplus.gov/cholesterollevelswhatyouneedtoknow.html):
   adult total-cholesterol and sex-specific HDL reference conventions. Full
   clinical context and a lipid profile are needed for interpretation.

The allocation of follow-up emphasis to LOW/MODERATE/HIGH is explicitly labelled
`PROJECT_WORKFLOW`. Guidelines support clinical evaluation and prevention, but
do not prescribe these model-probability strata or their relative follow-up
priority. In particular, "timely" is a conservative project workflow term, not a
guideline-defined deadline. The ADA principles do not establish validation of
this recommendation engine. Future guideline updates require human review and
a versioned rule change, not automatic changes to frozen model thresholds.

## Rule catalogue

| Rule ID | Trigger | Prompt category |
|---|---|---|
| REC_FOLLOWUP_LOW_001 | LOW | Routine preventive care and clinically indicated screening; no false reassurance |
| REC_FOLLOWUP_MODERATE_001 | MODERATE | Discuss result and whether screening/laboratory assessment is warranted |
| REC_FOLLOWUP_HIGH_001 | HIGH | Timely clinical review; discuss testing and clinician-directed referral when indicated |
| REC_SYMPTOMS_001 | All strata | Seek assessment for suggestive symptoms or other concerns regardless of score |
| REC_MONITOR_LOW_001 | LOW | Usual weight/BP/metabolic-health review |
| REC_MONITOR_MODERATE_001 | MODERATE | Broader cardiometabolic review and clinician-advised reassessment |
| REC_MONITOR_HIGH_001 | HIGH | Review testing history plus cardiometabolic measurements; no inferred glucose value |
| REC_ACTIVITY_001 | All | Activity suited to individual abilities and circumstances |
| REC_NUTRITION_001 | All | Individualized nutrient-dense eating support |
| REC_TOBACCO_001 | All | Avoid tobacco/vaping; maintain abstinence where applicable |
| REC_WEIGHT_001 | Observed age >=20 and BMI >=25 kg/m² | Discuss individualized weight-management support |
| REC_SMOKING_001 | Observed smoking_status=2 | Evidence-based cessation counseling/support |
| REC_BP_001 | Observed age >=20 and systolic >=120 OR diastolic >=80 mmHg | Confirm/review BP with a professional |
| REC_WAIST_001 | Observed age >=20; waist >101.6 cm with sex=1 or >88.9 cm with sex=2 | Contextual central-adiposity review |
| REC_LIPID_001 | Observed age >=20; total cholesterol >=200 OR HDL <40 with sex=1 / <50 with sex=2 mg/dL | Review full lipid profile |
| REC_MISSING_001 | Any original Core-10 input missing | Confirm missing measurements and expose imputation uncertainty |
| REC_SCOPE_001 | Age missing or <20 | Intended-population caution; adult measurement rules are not applied |

The waist values are exact conversions of NIDDK's strict >40-inch male and
>35-inch female conventions using 2.54 cm/inch; they are not rounded to 102/88 cm.
Sex-specific rules use the existing NHANES sex codes 1/2, which cannot capture
every individual's physiological or clinical context. If sex is missing, the
sex-specific conditions do not fire. An observed total cholesterol >=200 can
still independently trigger lipid review. Different populations, body composition,
pregnancy, and individual circumstances can require other interpretations; this
prototype does not establish these conventions as universally appropriate.

These clinical measurement review conventions are distinct from model-probability
boundaries. They do not change features, predictions, or eligibility. No new
clinical exclusion is imposed on the predictor. For known age below 20 or unknown
age, the existing prediction is preserved but a scope caution is added and adult
measurement-convention rules are skipped. Smoking support remains conditional
on the actual reported status. No race-specific treatment rule exists; race is
still a frozen predictor and may influence the probability indirectly.

Missing inputs never trigger measurement-abnormality rules via imputed values.
Partial BP information can trigger review if an observed component meets its
review convention. Physical activity is encouraged generally, not inferred from
Core-10 inputs; no physical-activity feature is added to the model. Similarly,
glucose/HbA1c history is something to discuss, not a new model input.

## Synthetic scenarios and audit trail

Run:

```powershell
.venv\Scripts\python.exe scripts/run_nhanes_risk_recommendation_scenarios.py
.venv\Scripts\python.exe -m streamlit run app.py
```

| Existing synthetic profile | Unchanged probability | Stratum | Follow-up / monitoring / lifestyle / feature-aware counts |
|---|---:|---|---|
| Relatively lower scoring | 0.005870712921023369 | LOW | 2 / 1 / 3 / 0 |
| Intermediate scoring | 0.06542942672967911 | LOW | 2 / 1 / 3 / 2 |
| Higher scoring | 0.5252653956413269 | HIGH | 2 / 1 / 3 / 5 |

All share REC_SYMPTOMS_001, REC_ACTIVITY_001, REC_NUTRITION_001, REC_TOBACCO_001.
The first two also trigger REC_FOLLOWUP_LOW_001 and REC_MONITOR_LOW_001. The
intermediate example additionally triggers REC_WEIGHT_001 and REC_BP_001.
The higher example triggers REC_FOLLOWUP_HIGH_001, REC_MONITOR_HIGH_001,
REC_WEIGHT_001, REC_SMOKING_001, REC_BP_001, REC_WAIST_001 and REC_LIPID_001.

The intermediate software label describes its relative score, not the new
MODERATE stratum. It is correctly LOW. No demonstration profile was changed or
optimized to populate MODERATE. MODERATE behavior and its follow-up prompts are
tested using explicit probability-only rule tests, never presented as a new
patient prediction.

Saved outputs:

- `results/nhanes/cdss/risk_recommendation_scenarios.csv`: scenario, probability,
  stratum, JSON list of triggered IDs, and all four section counts.
- `results/nhanes/cdss/risk_recommendation_provenance.json`: versions, dates,
  exact frozen references, source catalogue, scenario inputs/recommendations,
  pre/post input hashes and prior XAI hashes; flags for unchanged predictions and
  SHAP, no fitting, no threshold optimization and no cohort access.
- `tests/fixtures/nhanes_cdss_pre_recommendation_baseline.json`: the complete
  three synthetic outputs captured before this layer was added, linked to the
  frozen artifact SHA-256. It makes exact regression checks independent of the
  new layer. No real identifiers or patient records are stored.

The scenario runner compares **every pre-existing backend field exactly**, not
just rounded probability, and refuses to save a successful audit if it differs.
It also protects previous global/LIME explanation outputs by checksum.

## UI, testing and limitations

The UI order is input, submitted probability, model-derived stratum and disclaimer,
existing references/SHAP, four suggested-action tabs, model information and
limitations. Rule IDs/source labels appear with prompts; trigger descriptions,
rationales and framework version are expandable. LIME remains outside the normal
patient UI. No new predictor or duplicate prediction code is introduced.

Tests cover exact boundaries and adjacent floating-point values, the unused
Youden reference, stored threshold loading/order, correct labels, all four
sections, current/noncurrent smoking, observed-measurement boundaries, missing
inputs, age scope, deterministic actions independent of SHAP sign and race code,
JSON serialization, source traceability, UI delegation/display, scenario schemas,
and exact legacy probability/SHAP preservation. Fitting and cohort CSV access
are prohibited during the new framework tests. The legacy raw-cohort integration
module `tests/test_nhanes.py` is excluded from regression to avoid temporal reads.

Verification: **111 tests passed in 106.53 seconds**, including 33 new framework
cases and the seven updated prototype tests. The focused recommendation/backend
run passed 45 tests; the UI run passed seven. No runtime warnings were reported.
`pip check` found no broken requirements. Artifact and metadata hashes remained
unchanged, including artifact SHA-256
`fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`.
The successful scenario audit asserts exact preservation of every legacy result
field and does not read any cohort CSV. No model retraining or temporal-cohort
use occurred.

Objective 3 is implemented as a transparent academic CDSS framework. There is no
prospective validation of the strata, recommendation benefit, triage safety or
clinical workflow. Symptoms, prior diagnoses, medications, pregnancy, family
history, comorbidities and laboratory results are not collected comprehensively;
therefore rules cannot replace a clinician's assessment or a treatment plan.
The model's known missingness/generalization limitations remain. An abnormal
measurement or positive SHAP value is not proof of disease or an indication for
treatment. The tool neither diagnoses diabetes/hypertension/dyslipidemia nor
prescribes drugs, doses, insulin, surgery or individual treatment targets.

Major-project scope may include clinician-reviewed richer management context,
explicit guideline applicability, monitored updates, prospective workflow and
usability evaluation, and clinical validation. Those extensions would require
separate design and governance; none is implemented by this minor-project layer.

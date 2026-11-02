# Minor-project objective validation

## Scope, evidence levels and reproducibility

This is final integrated **software/prototype validation** of the feature-complete
minor-project implementation. It is not a new predictive experiment or prospective
clinical validation. The operational requirements supplied for this task define
the objective mapping; a verbatim approved synopsis document is not present in
the repository, so the descriptions below are requirement paraphrases rather
than invented verbatim objective quotations.

Evidence is separated into:

- **Implemented:** executable components exist and connect to the frozen artifact.
- **Experimentally evaluated:** saved development/temporal results from completed
  work are inspected without rerunning those research experiments.
- **Prototype-demonstrated:** synthetic end-to-end, repeated inference, Streamlit
  rendering and real localhost server-health checks pass.
- **Future clinical validation:** prospective clinician studies, real-world
  workflow effectiveness, diagnostic validity and deployment safety remain absent.

The machine-readable traceability matrix contains 21 requirements. COMPLETE
means complete for the stated implementation/evidence scope, not clinical
certification. PARTIAL records a substantive qualification; NOT IMPLEMENTED is
used when a claimed component or evidence file is missing. The final matrix is
not uniformly COMPLETE: the broad diagnosis/risk-prediction wording is qualified
by the diagnosed-status target, and LIME is qualified by limited local fidelity.

Run the final validation from the repository root:

```powershell
.venv\Scripts\python.exe scripts/validate_nhanes_cdss.py
```

This runs pytest, records a fresh run-linked Streamlit AppTest report, starts a
temporary headless Streamlit server on an unused localhost port, checks its health,
stops only that owned test process, and builds the audit bundle. The user's
normal port-8501 demonstration server is not stopped or reconfigured. Local socket
permission is needed inside the restricted Windows Codex sandbox.

The legacy `tests/test_nhanes.py` integration module is excluded because it reads
raw temporal data. All remaining tests run, including pre-existing unit tests
that use small synthetic estimator fits. No completed NHANES predictive experiment
is rerun and no frozen predictor/preprocessor is fitted. The new validation tests
prohibit frozen-system fitting and permit CSV reads only from an explicit list of
saved result tables. No raw temporal cohort is reopened.

## A. Original Objective 1 — prediction, comparison and evaluation

**Requirement.** Machine-learning diabetes diagnosis/risk support, model comparison
and performance evaluation.

**Implementation.** The verified `nhanes_phase2_xgb_v1` fitted pipeline performs
Core-10 classification. Previous Logistic Regression/XGBoost comparisons, nested
cross-validation, probability-quality work and separate temporal evaluation exist.
The final validator compares all fitted XGBoost hyperparameters, feature ordering,
pipeline step structure, imputer strategies/statistics, encoder categories and exact
stored thresholds against frozen metadata. Artifact/metadata hashes must also
match previous SHAP and recommendation provenance.

**Evidence.** Frozen artifact/metadata; saved baseline/nested summaries; temporal
metrics, threshold-transport and missingness tables; final integrity report.

| XGBoost evidence | ROC-AUC | Average precision | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Development nested CV | 0.826579 | 0.451996 | 0.098840 | Not requested in this summary |
| Separate temporal evaluation | 0.812354 | 0.424727 | 0.100014 | 0.325229 |

Full precision, CV standard deviations, LR comparison values, evidence paths and
hashes are retained in `final_validation_summary.json`. Temporal AUC decreased by
about 0.0142 and AP by 0.0273; Brier increased by about 0.00117 compared with the
saved nested estimate. These are retrospective performance comparisons, not a
new test on temporal participants.

At the frozen approximately 80%-sensitivity reference, temporal sensitivity was
0.792523 and specificity 0.664244. At the approximately 90% reference, they were
0.899065 and 0.521652. These names describe development operating points; they do
not guarantee sensitivity in a new population or within a stratum.

**Conclusion.** Classification/risk-support software, comparison and retrospective
evaluation are implemented. The broad diagnosis/future-risk interpretation remains
PARTIAL: DIQ010 physician-diagnosed status is not prospective incident diabetes or
a clinical diagnostic endpoint. No diagnostic-device or prospective-validity claim
is made.

## B. Original Objective 2 — factor identification and explainability

**Requirement.** Identify model-associated factors using global/local SHAP and
complementary LIME explanations.

**Implementation.** Existing exact Tree SHAP explains raw XGBoost margin/log-odds.
Signed one-hot contributions are aggregated to the ten original inputs before
ranking. Existing LIME uses development-reference perturbations to approximate
the frozen probability output. Neither method is changed or retuned here.

**Evidence.** Saved global SHAP tables/figures and provenance; current local
additivity and reconstructed-probability checks; saved LIME results and agreement.
The global SHAP maximum additivity error was 4.807909e-6. For the original three
synthetic profiles, SHAP–LIME top-3 overlap was 2/3, 2/3 and 3/3. Top-5 overlap was
4/5, 4/5 and 5/5. The saved primary LIME weighted R² values were approximately
0.392, 0.111 and 0.463; pointwise discrepancies were approximately 0.162, 0.063
and 0.098 on the probability scale. One surrogate output was negative. No LIME
computation is added for the new moderate demonstration.

**Conclusion.** SHAP factor identification and global/local explanations are
implemented and additivity-checked. LIME is implemented as a complement, but its
fidelity evidence is PARTIAL; feature overlap/stability does not make its surrogate
an exact account of the frozen prediction. SHAP remains primary. Explanations
describe model behavior, not causal disease mechanisms.

## C. Original Objective 3 — stratification and guidance

**Requirement.** Patient-specific Low/Moderate/High stratification with lifestyle,
monitoring and clinical-follow-up guidance.

**Implementation.** Existing exact sensitivity references define LOW below
0.08646801114082336, MODERATE from that lower boundary inclusive to
0.14026810228824615 exclusive, and HIGH at/above the latter. Labels always include
"model-derived". Seventeen central rules provide four sections: clinical follow-up,
monitoring, lifestyle and feature-aware considerations. Rules use observed inputs,
not SHAP direction or imputed abnormalities, with stable IDs and source traceability.
No rule content or boundary is changed during this validation.

**Evidence.** Existing config/provenance, rule/boundary tests, final four-scenario
table and normal UI rendering. Numeric review conventions and guideline sources
remain documented in `docs/methodology/nhanes_risk_recommendation_framework.md`.

**Conclusion.** Implemented and prototype-demonstrated, including all three strata
and smoking, weight, BP, waist and lipid review prompts. The strata and relative
follow-up emphasis are project-defined, not validated clinical cutoffs or an
ADA-endorsed model-probability triage policy. Recommendations remain educational,
rule-based and guideline-informed; clinicians determine applicability.

## D. Original Objective 4 — integrated and validated CDSS prototype

**Requirement.** Integrate input validation, prediction, stratum, explanations and
recommendations in a usable interface, with validation and explicit limitations.

**Implementation.** The existing Streamlit form delegates to the frozen backend.
Final validation follows input through the backend, raw probability, risk label,
SHAP, rules and presentation helpers. The application design and patient-facing
output are unchanged.

**Evidence and checks.**

- Four complete synthetic input profiles run through the backend three times each.
  Full JSON and presentation snapshots must match exactly, including probabilities,
  SHAP ordering, recommendation IDs/order/text and labels.
- Direct fitted-pipeline probability must equal the backend probability exactly.
  Reapplying the post-prediction layer must preserve all lower-layer fields without
  mutation. Every original result field for the three existing examples must match
  the captured pre-recommendation baseline exactly.
- Adjacent floating-point values around both boundaries, exact boundaries and an
  interior MODERATE value are checked. Missing required fields and invalid category
  codes fail clearly through the existing UI integration helper. Missing values
  remain distinct from missing fields: the backend supports imputation, while this
  UI requires all ten controls.
- Streamlit AppTest uses the actual backend for all four profiles, checks explicit
  submission, percentage/stratum display, SHAP chart/table, four action tabs,
  model information and disclaimers. Matplotlib canvas rendering checks plotted
  bar values against backend contributions. A separate real-server health response
  must be HTTP 200 with body `ok`.
- All configured recommendation texts and audit descriptions are screened, including
  untriggered rules, plus backend limitations and actual AppTest-rendered UI text.
  Required disclaimer concepts are checked in the rendered output.

**Conclusion.** Objective 4 is implemented and prototype-validated within the
supplied requirements. This verifies software integration and demonstration
readiness; it does not establish clinical effectiveness or prospective safety.

## Final synthetic demonstrations and evaluation procedure

| Software scenario | Frozen probability | Model-derived stratum | Feature-aware prompts |
|---|---:|---|---|
| Existing lower scoring | 0.005870712921023369 | LOW | None |
| Existing intermediate scoring | 0.06542942672967911 | LOW | Weight and BP |
| Existing higher scoring | 0.5252653956413269 | HIGH | Weight, current smoking, BP, waist and lipids |
| Moderate-stratum software demonstration | 0.0969625934958458 | MODERATE | Weight and BP |

The additional profile is one manually specified variant of the existing
intermediate example at age 60; every other value is unchanged. It was specified
once and evaluated without a model search or optimization. It is a software
demonstration, not a clinical archetype. The original intermediate remains LOW.

To reproduce the moderate example in the unchanged UI, load the intermediate
synthetic profile, change Age to 60, then press Run Diabetes Assessment. Show the
probability, mandatory model-derived label, SHAP chart and four action tabs.
For comparison, load and submit the three unchanged existing profiles. An edited
form does not itself generate a new prediction; results describe the last
submission. The ordinary demonstration launch command remains:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

## Clinical-language audit and interpretation

The new audit checks direct diagnostic assertions/denials, unsupported diagnosis
or clinical-validation wording, medication/dosage instructions and alarming or
guaranteeing terms. It includes adversarial unit tests and contextual allowances:
"physician-diagnosed" identifies the actual target; explicit denials of clinical
validity/prescribing are appropriate. Concentration units such as mg/dL are not
dosages. Documentation examples of prohibited wording are never supplied as
patient-facing audit input.

During development of the validator, one lexical false positive flagged the
existing coordinated disclaimer "do not diagnose disease, replace a treatment
plan, or prescribe medication." The auditor was corrected to respect this explicit
negation, with paired tests that still flag affirmative prescribing after "but".
No recommendation or UI wording was changed to obtain a passing audit.

Passing this bounded lexical/context audit establishes that the inspected strings
meet the tested language constraints. It cannot guarantee clinical safety, detect
every possible unsupported medical statement, or substitute for clinician review.
AppTest can emit a benign missing-ScriptRunContext harness log; actual app
exceptions/errors and test failures remain failures.

## Saved final audit bundle

Under `results/nhanes/cdss/`:

- `objective_traceability.csv`: all 21 mappings and qualified statuses.
- `final_end_to_end_scenarios.csv`: input definitions, probabilities, strata, SHAP
  directions, rules/counts, integrity errors, version and determinism outcomes.
- `clinical_language_audit.csv`: inspected strings, checks, context notes and flags.
- `final_validation_summary.json`: saved predictive evidence, threshold transport,
  explainability limitations, current integrity/boundary/smoke/test results.
- `final_cdss_validation_provenance.json`: source/evidence hashes before/after,
  artifact/metadata/application hashes, package versions, timestamp, test count,
  inputs, outputs and explicit no-retraining/no-experiment statements.
- `final_pytest_report.xml` and `final_pytest.log`: actual fresh regression evidence.
- `final_streamlit_smoke.json` and `final_streamlit_server.log`: run-linked actual
  UI observations, rendered text and temporary server health evidence.

There is no application release tag in the repository; the app source SHA-256
identifies the validated application instead of inventing a version number.

## Recorded final outcome

- **135 tests passed in 129.75 seconds**, zero failures/errors/skips among the
  executed tests. The legacy raw-temporal integration module was explicitly
  excluded; the 24 newly added integrated cases all passed.
- Real temporary Streamlit server: HTTP **200**, body **ok**. Its owned process
  tree was stopped after the check. AppTest verified all four profiles, required
  sections and disclaimers using the actual backend; no UI redesign occurred.
- Traceability: **19 COMPLETE, 2 PARTIAL, 0 NOT IMPLEMENTED**. The PARTIAL rows are
  OBJ1_PRED (target/diagnostic interpretation) and OBJ2_LIME (surrogate fidelity).
- Clinical-language audit: **232 checks PASS, zero flags**, including five required
  normal-UI disclaimer concepts. This is the bounded audit defined above, not a
  clinical safety certification.
- Every backend/presentation result matched exactly across three repetitions per
  profile. Existing three-profile outputs matched the pre-framework baseline;
  direct frozen probabilities matched the full backend for all four profiles.
- Maximum current local SHAP margin reconstruction error: **1.655425876379013e-6**.
  Maximum reconstructed-probability error: **1.3878478469231406e-7**.
- All protected component and historical-evidence hashes matched before/after.
  Artifact SHA-256 remains
  `fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`;
  metadata SHA-256 remains
  `57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3`.
- No frozen-model retraining, preprocessing refitting, threshold/rule changes,
  altered SHAP/LIME implementation, new predictive experiment, raw temporal-cohort
  access, or patient-facing output change occurred.

## Remaining limitations and feature-completeness conclusion

1. The target is physician-diagnosed diabetes status, not future onset or an
   independently adjudicated diagnostic endpoint. The model is not a diagnostic
   medical device.
2. NHANES population/cohort selection may not generalize to every setting or
   population. The saved temporal results show a modest decrease in discrimination
   and slightly worse probability losses compared with development.
3. Missing predictor measurements were associated with weaker temporal performance:
   XGBoost AUC was about 0.8314 with all Core-10 observed versus 0.7712 with one or
   more missing. This is an association, not an estimate of a causal missingness effect.
4. Strata are project-defined decision-support interpretations of development
   operating references, not diagnostic or clinically validated risk categories.
5. Recommendations are rule-based, guideline-informed prompts and omit much
   clinical history, symptoms, laboratory context, pregnancy and comorbidity data.
   No medication selection, dosing or individualized treatment plan is produced.
6. SHAP is a model explanation, not causation; LIME has material approximation
   errors despite reproducible feature sets.
7. There is no EHR integration, prospective clinician study, real-world deployment
   validation or evidence of improved patient outcomes.
8. The local prototype lacks production clinical governance, authentication and
   deployment controls; no regulatory approval is claimed.

The minor-project software is feature-complete for the supplied operational scope.
Objectives 1–4 are implemented, with retrospective experimental evidence and
prototype validation as specified above. The matrix retains explicit PARTIAL
qualifications rather than equating implementation with diagnostic validity or
exact LIME fidelity. Future clinical validation and richer management workflows
remain outside this minor-project completion claim.

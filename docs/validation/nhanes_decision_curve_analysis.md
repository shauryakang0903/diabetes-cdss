# NHANES decision curve analysis

**SECONDARY population-level analysis. Temporal results are Temporal post-hoc DCA /
POST-HOC / SECONDARY ANALYSIS after previous evaluation of the 2021–2023 cohort.**

## Purpose and decision interpretation

Decision curve analysis (DCA) compares the empirical net benefit of using model
probabilities to guide assessment/referral with evaluating everyone or no one.
It incorporates a specified trade-off between identifying a target-positive
participant and an unnecessary evaluation of a target-negative participant. This
is a decision-analytic calculation, not a clinical trial, treatment-effect study,
prospective validation or demonstration of improved patient outcomes.

The approach follows [Vickers and Elkin (2006), Decision Curve Analysis: A Novel
Method for Evaluating Prediction Models](https://doi.org/10.1177/0272989X06295361).
For the role of threshold preferences and clinical context, see
[Vickers, van Calster and Steyerberg (2019), A simple, step-by-step guide to
interpreting decision curve analysis](https://doi.org/10.1186/s41512-019-0064-7).
Here the decision is described as **assessment/referral**, not treatment allocation.

## Formula and benchmarks

At threshold probability pt, define positive as predicted probability >= pt:

```text
Net benefit = TP/N - FP/N * pt/(1-pt)
Evaluate-none net benefit = 0
Evaluate-all net benefit = prevalence - (1-prevalence) * pt/(1-pt)
```

The threshold odds pt/(1-pt) specify the false-positive evaluation penalty relative
to the value of correctly identifying a target-positive participant. A lower pt
represents greater willingness to evaluate to avoid missed positives; it is not a
proposed threshold selected by this analysis. Net benefit uses target-positive
equivalents per participant and can be negative. It is not a probability or a
measured number of improved patient outcomes.

The formulas are implemented directly from confusion counts. Probabilities and
thresholds must be finite; thresholds <=0 or >=1 are rejected. Cases equal to a
threshold count as positive. Evaluate-all/none-identical model classifications
are explicitly identical in floating point. Benchmark superiority uses an NB
tolerance of 1e-12 solely to avoid numerical false superiority at equality; this
is neither a clinical relevance margin nor a statistical equivalence test.

Do not choose a threshold by finding the largest plotted NB across the x-axis:
changing pt also changes the utility weight. Strategies are compared **at the
same preference**, not as a threshold-optimization exercise.

## Prediction sources and integrity

### Development OOF

Use the existing `results/nhanes/threshold_oof_predictions.csv` for XGBoost and
Logistic Regression, with N=25,580 and 3,690 positives (prevalence
0.14425332290852227). There are no full-development in-sample predictions or new
fits. The existing subgroup-audit loader verifies the development CSV and OOF
hashes, unique participant coverage, target/cycle labels, archived XGBoost
parameters and saved stratified 5-fold membership (shuffle=True, seed 42).

LR rows are aligned to the verified XGBoost participant IDs, cycles, targets and
folds, then to development-row order. Shuffling prediction rows cannot change the
result. Missing, duplicate or mismatched comparator rows fail; they are not
silently joined by position. The archived LR configuration includes C=0.1,
lbfgs, max_iter=2000 and no class weighting; its full parameter representation is
retained in the DCA summary. No LR object is constructed or fitted.

These are fixed-configuration OOF probabilities selected after earlier development
work, not a new fully nested DCA estimate. They avoid using a participant's own
training-fit prediction but do not remove earlier configuration-selection
uncertainty or correlations among shared training-fold fits.

### Temporal post-hoc

Reapply the existing verified `phase2_xgboost_pipeline.joblib` through the existing
frozen probability helper to the already-evaluated 2021–2023 cohort: N=7,536,
1,070 positives, prevalence 0.14198513800424628. The temporal CSV hash is checked
against previous evaluation provenance. No fitting, recalibration, model change,
new dataset or temporal tuning occurs.

Repository search found no saved temporal participant-level LR probabilities and
no frozen LR artifact. Therefore temporal LR is **omitted**, not refitted or
approximated. The temporal CSV includes an unavailable/empty `lr_net_benefit`
column and `lr_available=False`; figures and reference rows include XGBoost only.
The development SHAP NPZ contains full-development XGBoost explanation data and
is not substituted for OOF probabilities or treated as a comparator source.

## Pre-specified threshold grid and frozen references

The primary grid is 0.01–0.50 inclusive in steps of 0.005, constructed with integer
increments divided by 1,000 (99 base points). Insert the exact four metadata
values below; default 0.5 already exists in the base grid. This gives **102 unique
tested thresholds**:

| Metadata key | Exact value |
|---|---:|
| sensitivity_90_reference | 0.08646801114082336 |
| youden_reference | 0.12872706353664398 |
| sensitivity_80_reference | 0.14026810228824615 |
| default | 0.5 |

Every model is evaluated at the same pt, including the XGBoost-derived reference
values when comparing development LR. These are not LR-specific sensitivity
targets. Values are loaded from frozen metadata rather than redefined/rounded.
Figures draw lines at their exact positions; captions round displayed labels only.

The references were defined previously from development operating characteristics.
They were **not elicited from clinician/patient assessment preferences**. Likewise,
the requested broad 0.01–0.50 range is an analysis range, not a declaration that
all of it is clinically reasonable for a specified referral pathway. No threshold
is selected, optimized or changed by DCA.

## Results at frozen reference thresholds

These are point estimates, without DCA confidence intervals or significance tests.
Full precision, confusion counts, sensitivity/specificity and both benchmarks
are stored in `dca_reference_thresholds.csv` and `dca_summary.json`.

| Frozen reference | Development XGBoost NB | Development LR NB | Temporal post-hoc XGBoost NB |
|---|---:|---:|---:|
| 90%-sensitivity reference | 0.09437588696828067 | 0.09255187258077059 | 0.08880574215813195 |
| Youden reference | 0.07734830343084367 | 0.07399980883230611 | 0.07010919294592094 |
| 80%-sensitivity reference | 0.07251591725544279 | 0.06993613745044536 | 0.06552470413932351 |
| Default 0.5 | 0.010164190774042223 | 0.005707584050039093 | 0.007032908704883226 |

| Reference | Development evaluate-all NB | Temporal evaluate-all NB | Evaluate-none NB |
|---|---:|---:|---:|
| 90% | 0.06325483121818372 | 0.060771957129550525 | 0 |
| Youden | 0.0178202015948091 | 0.01521690151586605 | 0 |
| 80% | 0.004635422543798978 | 0.001997175771388915 | 0 |
| Default 0.5 | -0.7114933541829553 | -0.7160297239915074 | 0 |

At the frozen 80% reference, development XGBoost has TP=2,952, FP=6,724,
TN=15,166, FN=738. Temporal post-hoc XGBoost has TP=848, FP=2,171, TN=4,295,
FN=222. These counts allow direct verification of the formula and do not measure
downstream treatment benefit.

## Favorable and unfavorable sampled-grid runs

- Development XGBoost: above both evaluate-all and evaluate-none at all **102/102**
  tested values, one sampled-grid run from **0.01 to 0.50**.
- Temporal post-hoc XGBoost: also above both at **102/102**, sampled-grid run
  **0.01 to 0.50**.
- Development LR: above both benchmarks at **102/102**, but XGBoost has higher NB
  than LR at every tested point. No temporal LR comparison is available.
- XGBoost and development LR have **no tested unfavorable or tied points** against
  either benchmark within the specified grid. All such interval arrays are saved
  explicitly as empty, rather than omitted. This is not a claim about thresholds
  outside the range or every untested value between adjacent grid points.

The reusable interval extractor preserves each disjoint run, including single
points, and records neighboring tested thresholds. It does not estimate continuous
crossing locations. Tests exercise separated favorable regions and empty runs.
No favorable subrange was chosen for exclusive display, and no unfavorable model
point was removed. Evaluate-all's negative NB region is fully retained.

## Figures

Matplotlib-only exports, each 300-dpi PNG and vector PDF with embedded TrueType
fonts:

- `figures/development_dca.png` / `.pdf`: full development OOF range, XGBoost, LR,
  evaluate-all and evaluate-none.
- `figures/temporal_dca.png` / `.pdf`: full temporal post-hoc range, XGBoost and
  both benchmarks; no fabricated LR curve.
- `figures/development_dca_zoomed.png` / `.pdf` and
  `figures/temporal_dca_zoomed.png` / `.pdf`: explicitly labelled secondary
  0.05–0.20 views, supplied alongside—not instead of—the full-range figures.

The full plots include all negative evaluate-all values, which compresses the
positive model curves. The fixed zoom improves readability while preserving the
full-range evidence. Reference labels are placed in captions rather than on top
of the curves; the three subtle vertical lines retain exact reference positions.

## Subgroup caution and methodological limitations

See the completed [post-hoc subgroup audit](nhanes_subgroup_performance_audit.md).
At the frozen 80% reference, temporal sensitivity was 12.5% in ages 20–39 versus
approximately 91.47% in ages 60+. Population-level DCA **does not resolve** this
threshold-performance variation or establish equal decision benefit across
demographic groups. No subgroup DCA, subgroup threshold optimization, model change
or rule change is introduced in response.

Important limits on interpretation:

1. **Endpoint/action mismatch.** The endpoint is physician-diagnosed diabetes
   status, not future incident diabetes, biochemical disease ascertainment or
   independently validated need for assessment/referral. Previously diagnosed
   people may already be receiving care; DIQ010-negative people can include
   undiagnosed disease. Consequently a target false positive is not necessarily
   an unnecessary real-world evaluation. This limits clinical interpretation even
   when the mathematical DCA is correct for the study labels.
2. **Hypothetical preferences.** The pt odds impose a specified false-positive
   penalty. Neither the frozen sensitivity references nor the entire grid was
   established as a clinical preference distribution. A real assessment pathway,
   its harms, costs, burdens and patient priorities are not validated here.
3. **No prospective benefit claim.** DCA does not observe downstream diagnostic
   testing, care changes, treatment outcomes or treatment effects. There is no
   prospective clinician evaluation or clinical trial.
4. **Post-hoc exposure.** Temporal participants were evaluated previously. These
   secondary results are not a new independent temporal validation experiment.
5. **Uncertainty and sampling.** Results are unweighted population/cohort-level
   point estimates, not NHANES complex-survey national estimates. No confidence
   intervals, statistical-superiority tests or fitting/tuning-uncertainty analysis
   are provided. Prior model selection and imperfect probability calibration can
   influence empirical net benefit; neither model is recalibrated here.
6. **Comparator availability.** Development LR uses valid archived OOF data;
   absence of temporal LR prevents a temporal model-to-model DCA claim.

No computational or input-integrity issue was found that invalidates these
empirical curves under the stated assumptions. However, interpreting them as
actual clinical benefit, proof of suitable referral thresholds or equal subgroup
utility would be unsupported without resolving the endpoint/preference/context
issues above. They describe potential decision-support value under hypothetical
decision assumptions, not deployment readiness.

## Code, outputs, integrity and reproduction

New files:

- `src/evaluation/nhanes_decision_curve.py`
- `scripts/run_nhanes_decision_curve.py`
- `tests/test_nhanes_decision_curve.py`
- This methodology document.

Saved under `results/nhanes/decision_curve/`:

- `development_dca.csv`, `temporal_dca.csv`: 102 rows each, standard NB curves,
  benchmark curves, confusion counts, availability and numerical comparison flags.
- `dca_reference_thresholds.csv`: 12 model/dataset/reference rows at exact values.
- `dca_summary.json`: prevalence/N, every favorable/unfavorable/tied sampled-grid
  run, reference results, comparator provenance, fixed settings, warnings, limitations,
  package/source versions, timestamps and input/output integrity hashes.
- Eight PNG/PDF figure exports, and the independently generated full-suite XML report.

Standardized NB is not included, keeping the requested standard NB primary. DCA
is deterministic conditional on the saved probabilities; it uses no new random
sampling. SHA-256 checks protect the artifact, metadata, cohorts, OOF predictions,
previous evidence and existing model/application/rule components before/after.
Only outputs owned by this runner are included in its output hash manifest, so
regenerating an independent test report cannot invalidate analysis hashes.

```powershell
.venv\Scripts\python.exe scripts/run_nhanes_decision_curve.py
.venv\Scripts\python.exe -m pytest tests/test_nhanes_decision_curve.py -q
.venv\Scripts\python.exe -m pytest -q
```

The existing Streamlit tests need local socket permission within the restricted
Windows sandbox. The full suite includes read-only raw-cohort integration tests;
this is consistent with explicitly post-hoc temporal access. Frozen-model fitting
and preprocessing fitting are prohibited during the new DCA tests.

Verification: the focused DCA suite passed **16 tests**; the complete suite passed
**184 tests in 83.45 seconds**, with zero failures/errors/skips. The focused suite
was also rerun after the final figure-label cleanup and explicit LR fitting guard.
Tests cover direct hand calculations, exact benchmark identities, threshold
validation, deterministic curves, disjoint/empty interval runs, exact frozen
references, shuffled/mismatched OOF IDs, frozen-only temporal inference, output
schemas and all eight figure exports. XGBoost, LR, pipeline and preprocessing fit
methods are blocked during the new tests.

Artifact SHA-256 **before = after**:
`fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528`.

Metadata SHA-256 **before = after**:
`57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3`.

All current input, protected component, previous-result, source and owned-output
hashes were checked against the saved summary after execution. No runtime warnings
were recorded. The model artifacts, metadata, thresholds, preprocessing, rules and
Streamlit source were unchanged. No new threshold or subgroup threshold was selected.

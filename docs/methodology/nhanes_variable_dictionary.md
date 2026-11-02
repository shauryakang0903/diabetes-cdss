# NHANES Phase-2 Variable Dictionary

## Dataset

NHANES 2017–March 2020 Pre-Pandemic Public-Use Dataset

## Merge Key

| Variable | File | Meaning | Role |
|---|---|---|---|
| SEQN | All component files | Respondent sequence number | Merge key only |

---

## Outcome

| Variable | File | Meaning | Coding / Unit | Proposed Role |
|---|---|---|---|---|
| DIQ010 | P_DIQ | Doctor/health professional ever told participant they have diabetes or sugar diabetes, excluding pregnancy-related diabetes | 1 = Yes, 2 = No, 3 = Borderline, 7 = Refused, 9 = Don't know | Primary outcome candidate |

### Initial outcome handling

For the first model:

- `DIQ010 = 1` → diabetes positive
- `DIQ010 = 2` → diabetes negative
- `DIQ010 = 3` → do not automatically classify yet; handle separately during cohort design
- `7`, `9`, missing → exclude from supervised outcome

Important: this is a self-reported physician-diagnosed diabetes outcome.

---

## Demographic Predictors

| Variable | File | Meaning | Unit / Coding | Proposed Role |
|---|---|---|---|---|
| RIDAGEYR | P_DEMO | Age at screening | Years; age 80+ top-coded at 80 | Predictor |
| RIAGENDR | P_DEMO | Sex/gender recorded for participant | Categorical | Predictor |
| Race/ethnicity variable | P_DEMO | Race/ethnicity category | Categorical | Pending exact variable selection |

Adults aged `< 20` will be excluded from the Phase-2 analytical cohort.

---

## Anthropometric Predictors

| Variable | File | Meaning | Unit | Proposed Role |
|---|---|---|---|---|
| BMXBMI | P_BMX | Body Mass Index | kg/m² | Predictor |
| BMXWAIST | P_BMX | Waist circumference | cm | Predictor |
| BMXWT | P_BMX | Body weight | kg | Candidate predictor |
| BMXHT | P_BMX | Standing height | cm | Candidate / possibly redundant |

BMI and waist circumference will likely be preferred over simultaneously using weight, height and BMI unless feature analysis justifies them.

---

## Blood Pressure Predictors

Source file: `P_BPXO`

NHANES recorded three consecutive oscillometric blood-pressure measurements.

| Variable | Meaning | Proposed Role |
|---|---|---|
| BPXOSY1 | Systolic BP, reading 1 | Intermediate |
| BPXOSY2 | Systolic BP, reading 2 | Intermediate |
| BPXOSY3 | Systolic BP, reading 3 | Intermediate |
| BPXODI1 | Diastolic BP, reading 1 | Intermediate |
| BPXODI2 | Diastolic BP, reading 2 | Intermediate |
| BPXODI3 | Diastolic BP, reading 3 | Intermediate |

We will derive:

- Mean systolic blood pressure
- Mean diastolic blood pressure

from valid readings rather than simply choosing one measurement.

---

## Glycemic Laboratory Variables

### HbA1c

| Variable | File | Meaning | Unit | Proposed Role |
|---|---|---|---|---|
| LBXGH | P_GHB | Glycohemoglobin / HbA1c | % | Candidate predictor / alternative outcome-definition variable |

### Fasting Plasma Glucose

| Variable | File | Meaning | Unit | Proposed Role |
|---|---|---|---|---|
| LBXGLU | P_GLU | Fasting plasma glucose | mg/dL | Candidate predictor / alternative outcome-definition variable |
| WTSAFPRP | P_GLU | Fasting subsample weight | Survey-analysis variable | Not a predictive feature |

Important:

Fasting glucose is measured only in the NHANES fasting subsample, so inclusion of `LBXGLU` can substantially reduce the usable analytical cohort.

---

## Lipid Variables

| Variable | File | Meaning | Unit | Proposed Role |
|---|---|---|---|---|
| LBDHDD | P_HDL | Direct HDL cholesterol | mg/dL | Candidate predictor |
| LBXTC | P_TCHOL | Total cholesterol | mg/dL | Candidate predictor |
| LBDLDL | P_TRIGLY | LDL cholesterol calculated using Friedewald equation | mg/dL | Candidate predictor |
| LBDLDLM | P_TRIGLY | LDL cholesterol calculated using Martin-Hopkins equation | mg/dL | Candidate predictor |

We will not blindly include every lipid variable. Multicollinearity and missingness will be assessed first.

---

## Smoking Variables

Source file: `P_SMQ`

| Variable | Meaning | Proposed Role |
|---|---|---|
| SMQ020 | Smoked at least 100 cigarettes during lifetime | Candidate predictor |
| SMQ040 | Current smoking frequency: every day / some days / not at all | Candidate predictor |

For adults, smoking status can later be derived into a cleaner feature such as:

- Never smoker
- Former smoker
- Current smoker

NHANES documentation indicates that current smoking status should be derived using the lifetime-smoking and current-smoking questions together.

---

## Physical Activity Variables

Source file: `P_PAQ`

The adult physical-activity section contains questionnaire variables beginning with `PAQ605` through `PAQ680`.

Exact physical-activity features will be selected only after inspecting:

- variable definitions,
- skip patterns,
- missingness,
- interpretability,
- and compatibility with the intended CDSS.

Do not add all activity variables automatically.

---

# Variables That Must NOT Be Used Blindly

## SEQN

Participant ID only.

Never use as a model feature.

## Survey weights

Examples include variables such as:

- examination weights,
- interview weights,
- fasting subsample weights.

These describe the NHANES sampling design.

They must not be treated as biological predictors.

They may be required for population-representative statistical analyses.

## Outcome-derived diabetes variables

Variables such as:

- diabetes medication use,
- insulin use,
- age at diabetes diagnosis,
- diabetic complications,

can reveal that the person is already known to have diabetes.

These could create severe target leakage in a model whose outcome is existing diabetes diagnosis.

They must therefore be excluded from the predictive feature set unless we explicitly formulate a different research question.

---

# Critical Target-Leakage Decision

We will eventually evaluate at least two possible modeling formulations.

## Model formulation A — Known diabetes classification

Outcome:

`DIQ010`

Predictors could potentially include:

- Age
- Sex
- BMI
- Waist circumference
- Blood pressure
- Lipids
- Smoking
- Physical activity

Using HbA1c and fasting glucose here requires careful interpretation because they are themselves diagnostic biomarkers for diabetes.

## Model formulation B — Laboratory-defined diabetes

Outcome derived from laboratory criteria such as HbA1c and/or fasting glucose.

In that formulation:

- HbA1c used to define the target cannot simultaneously be used as a predictor.
- Fasting glucose used to define the target cannot simultaneously be used as a predictor.

This formulation could allow investigation of previously undiagnosed diabetes but requires a separate clinical definition and methodology.

---

# Initial Core Feature Set

The first candidate non-leaky feature set is:

1. Age
2. Sex
3. Race/ethnicity
4. BMI
5. Waist circumference
6. Mean systolic BP
7. Mean diastolic BP
8. HDL cholesterol
9. Total cholesterol
10. Smoking status
11. Physical-activity measure(s)

Potential laboratory-enhanced features:

12. HbA1c
13. Fasting glucose

These will only be included after the target formulation is finalized.

---

# Current Status

Verified from official NHANES documentation:

- P_DEMO
- P_DIQ
- P_BMX
- P_BPXO
- P_GHB
- P_GLU
- P_HDL
- P_TCHOL
- P_TRIGLY
- P_SMQ
- P_PAQ

Still required:

- exact race/ethnicity variable selection
- exact physical-activity feature construction
- outcome-definition decision
- laboratory versus non-laboratory model definition
- missingness audit
- final feature inclusion criteria
- survey-weight analysis strategy
# NHANES Cycle Harmonization

## Cycles Included

- 2011–2012
- 2013–2014
- 2015–2016
- 2017–March 2020 Pre-Pandemic

## Purpose

Create one harmonized adult NHANES cohort using variables that have
comparable definitions across survey cycles.

---

## Core Variable Mapping

| Concept | 2011–2012 | 2013–2014 | 2015–2016 | 2017–Mar 2020 | Harmonization |
|---|---|---|---|---|---|
| Participant ID | SEQN | SEQN | SEQN | SEQN | Direct |
| Age | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | Direct |
| Sex | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | Direct |
| Race/ethnicity | RIDRETH3 | RIDRETH3 | RIDRETH3 | RIDRETH3 | Direct |
| Diabetes diagnosis | DIQ010 | DIQ010 | DIQ010 | DIQ010 | Direct |
| BMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI | Direct |
| Waist circumference | BMXWAIST | BMXWAIST | BMXWAIST | BMXWAIST | Direct |
| HDL cholesterol | LBDHDD | LBDHDD | LBDHDD | LBDHDD | Direct |
| Total cholesterol | LBXTC | LBXTC | LBXTC | LBXTC | Direct |
| Lifetime smoking | SMQ020 | SMQ020 | SMQ020 | SMQ020 | Direct |
| Current smoking | SMQ040 | SMQ040 | SMQ040 | SMQ040 | Direct |
| Vigorous work activity | PAQ605 | PAQ605 | PAQ605 | PAQ605 | Direct |
| Moderate work activity | PAQ620 | PAQ620 | PAQ620 | PAQ620 | Direct |
| Walking/bicycling | PAQ635 | PAQ635 | PAQ635 | PAQ635 | Direct |
| Vigorous recreation | PAQ650 | PAQ650 | PAQ650 | PAQ650 | Direct |
| Moderate recreation | PAQ665 | PAQ665 | PAQ665 | PAQ665 | Direct |

---

## Blood Pressure Mapping

### 2011–2012
File: BPX_G.xpt

- BPXSY1
- BPXSY2
- BPXSY3
- BPXDI1
- BPXDI2
- BPXDI3

### 2013–2014
File: BPX_H.xpt

- BPXSY1
- BPXSY2
- BPXSY3
- BPXDI1
- BPXDI2
- BPXDI3

### 2015–2016
File: BPX_I.xpt

- BPXSY1
- BPXSY2
- BPXSY3
- BPXDI1
- BPXDI2
- BPXDI3

### 2017–March 2020
File: P_BPXO.xpt

- BPXOSY1
- BPXOSY2
- BPXOSY3
- BPXODI1
- BPXODI2
- BPXODI3

### Harmonized Derived Variables

All cycles will be converted into:

- mean_systolic_bp
- mean_diastolic_bp

The mean will be calculated using available valid readings.

Important:
The 2017–March 2020 cycle used oscillometric BP measurement, while earlier
cycles used the previous NHANES BP protocol. Therefore blood pressure is
conceptually harmonized, but measurement methodology differs by cycle.

A cycle indicator will be retained so that cycle-related differences can
later be evaluated.

---

## File Mapping

### 2011–2012

- DEMO_G.xpt
- DIQ_G.xpt
- BMX_G.xpt
- BPX_G.xpt
- HDL_G.xpt
- TCHOL_G.xpt
- SMQ_G.xpt
- PAQ_G.xpt

### 2013–2014

- DEMO_H.xpt
- DIQ_H.xpt
- BMX_H.xpt
- BPX_H.xpt
- HDL_H.xpt
- TCHOL_H.xpt
- SMQ_H.xpt
- PAQ_H.xpt

### 2015–2016

- DEMO_I.xpt
- DIQ_I.xpt
- BMX_I.xpt
- BPX_I.xpt
- HDL_I.xpt
- TCHOL_I.xpt
- SMQ_I.xpt
- PAQ_I.xpt

### 2017–March 2020

- P_DEMO.xpt
- P_DIQ.xpt
- P_BMX.xpt
- P_BPXO.xpt
- P_HDL.xpt
- P_TCHOL.xpt
- P_SMQ.xpt
- P_PAQ.xpt

---

## Standard Harmonized Names

After each cycle is loaded, variables will be renamed to common internal names:

- seqn
- age
- sex
- race_ethnicity
- diabetes_status
- bmi
- waist_cm
- mean_systolic_bp
- mean_diastolic_bp
- hdl_mg_dl
- total_cholesterol_mg_dl
- smoking_lifetime
- smoking_current
- vigorous_work
- moderate_work
- walk_bicycle
- vigorous_recreation
- moderate_recreation
- nhanes_cycle

---

## Important Harmonization Rules

1. Do not concatenate raw files directly.

2. Each cycle must first be cleaned and converted to the common internal
   variable names.

3. Outcome coding must be standardized before pooling.

4. Refused and "don't know" responses must be converted to missing values.

5. Physical-activity follow-up missing values must be interpreted according
   to questionnaire skip logic, not automatically treated as ordinary
   missing observations.

6. Blood-pressure readings must first be converted to harmonized mean
   systolic and diastolic variables.

7. Survey cycle must remain in the dataset as `nhanes_cycle`.

8. Original raw XPT files must never be modified.

9. Survey weights and design variables will be retained separately for
   epidemiological / population-representative analyses but will not be used
   as predictive patient features.

10. No modeling begins until the pooled cohort has passed a new missingness,
    class-balance, range, and duplicate audit.

---

## Current Status

2017–March 2020 cohort:
- total demographics participants: 15,560
- adults age >=20: 9,232
- valid DIQ010 outcome: 8,965
- diabetes positive: 1,420
- diabetes negative: 7,545
- complete selected continuous-variable cases: 6,748

Earlier cycles:
- variable mapping verified
- raw files not yet downloaded
- pooled cohort not yet constructed
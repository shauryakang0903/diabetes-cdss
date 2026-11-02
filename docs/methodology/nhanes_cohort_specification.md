# NHANES Phase-2 Cohort Specification

## Study Objective

Develop and evaluate an explainable machine-learning model for diabetes
classification/risk assessment in adults using NHANES.

## Primary Dataset

NHANES 2017–March 2020 Pre-Pandemic public-use dataset.

## Study Population

Include:
- Adults aged >= 20 years
- Participants with valid diabetes outcome information
- Participants with sufficient predictor information for modeling

Exclude:
- Participants younger than 20 years
- Participants with unknown/refused diabetes outcome
- Records lacking essential analysis variables

## Participant Identifier

SEQN

All NHANES component files will be merged using SEQN.

## Initial Outcome Candidate

DIQ010:
"Have you ever been told by a doctor or health professional that you have
diabetes or sugar diabetes?"

Outcome coding will be verified from the official NHANES codebook before
implementation.

## Initial Candidate Predictors

### Demographics
- Age
- Sex
- Race/ethnicity

### Anthropometric
- BMI
- Waist circumference

### Clinical
- Blood pressure measurements

### Laboratory
- HbA1c
- Glucose
- Lipid-related variables

### Lifestyle / Medical History
- Smoking
- Physical activity
- Hypertension history
- Cholesterol history

## Critical Leakage Rule

Variables used to define the outcome must not automatically also be used as
predictors.

For example, if HbA1c or fasting glucose is used to construct a laboratory
definition of diabetes, those same values cannot simply be included as input
features in that model without explicitly redefining the research question.

## Modeling Strategy

Models:
- Logistic Regression
- XGBoost

Evaluation:
- ROC-AUC
- PR-AUC
- Sensitivity
- Specificity
- Precision
- F1 score
- Brier score
- Calibration curves

Explainability:
- Logistic coefficients / odds ratios
- SHAP
- XAI stability analysis

## Validation

- Development/training cohort
- Cross-validation within development data
- Completely held-out test cohort

Survey-cycle-aware or temporal validation will be investigated after the
initial cohort has been constructed.

## Status

Dataset specification in progress.
No model training should begin until outcome, predictors, missingness and
cycle compatibility have been verified.
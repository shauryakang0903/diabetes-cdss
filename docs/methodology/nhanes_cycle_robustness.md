# Leave-one-development-cycle-out robustness

## Scope, fixed inputs, and split design

The only cohort input is `data/processed/nhanes/nhanes_development_2011_2020.csv`:
25,580 adults with valid physician-diagnosed diabetes outcomes from DIQ010,
3,690 positive and 21,890 negative. The shared loader validates frozen counts,
development cycle labels, adult ages, binary outcomes, and unique participant
IDs. No actual 2021-2023 temporal-validation file is read, inspected, summarized,
or even hashed. Before/after hashes verify unchanged development data and the
prior threshold file. Previous experiment outputs and raw XPTs are preserved.

The existing cycle field is named **nhanes_cycle**, not cycle. The four test
sets are 2011_2012, 2013_2014, 2015_2016, and 2017_2020 (2017-March 2020).
Exactly one entire cycle is held out per split. A fresh fixed LR and fixed
XGBoost pipeline train on the other three cycles. Training/test SEQN sets must
be disjoint and both sets must contain both outcome classes. All four cycles
must be present. There is no random row split or internal hyperparameter search.
For earlier held-out cycles, training includes later development cycles;
this is internal leave-cycle-out robustness, not chronological forecasting or
external/temporal validation.

## Frozen features, preprocessing, and estimators

Core-10 is unchanged: age, sex, race_ethnicity, bmi, waist_cm,
mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl, total_cholesterol_mg_dl,
smoking_status. The seven numeric features use median imputation, with
StandardScaler for LR only. Sex, race/ethnicity, and smoking status use
most-frequent imputation and OneHotEncoder(handle_unknown="ignore"). Each
preprocessing step is inside the full pipeline, fitted on training cycles
only. No full-cohort preprocessing occurs.

LR: L2 (l1_ratio=0 under installed sklearn 1.9), lbfgs, C=0.1,
max_iter=2000, class_weight=None, random_state=42. XGBoost: 200 trees,
learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8,
colsample_bytree=0.8, reg_alpha=1, reg_lambda=10, binary:logistic objective,
hist trees on CPU, gamma=0, scale_pos_weight=1, eval_metric=logloss,
n_jobs=1, random_state=42, no early stopping. Other installed-version defaults
are recorded with full model parameters and package versions in provenance.
The missing-value NaN model-parameter sentinel is encoded as the string "NaN"
for strict JSON. Existing raw pipeline construction is reused; no calibration
wrapper is called or modified. No new dependencies were installed.

## Previously selected thresholds: application only

Four points per model are loaded from
`results/nhanes/threshold_candidate_points.csv`, using
pd.read_csv(..., float_precision="round_trip"):

- default_0.50;
- nearest_sensitivity_0.80;
- nearest_sensitivity_0.90;
- youden_j_maximum.

The equality-of-sensitivity/specificity reference is not part of this experiment.
Each model/reference must occur exactly once. Saved default values must be 0.5,
saved sensitivity targets must be 0.8/0.9, and thresholds must be finite and in
[0,1]. Full Python-float threshold values are preserved in CSV outputs and
provenance. No rounded threshold is substituted and no threshold-selection
function is called. Predictions are positive when probability >= saved threshold.
Every reference is applied unchanged to all four held-out cycles.

For each held-out cycle, sensitivity_target_deviation = achieved sensitivity
minus the intended 0.8 or 0.9; absolute_sensitivity_target_deviation is its
absolute value. These fields are missing for default/Youden because those
references do not prescribe a sensitivity target. Deviation values are rates;
multiply by 100 for percentage points. Targets are descriptive development
references, not guaranteed screening sensitivity or clinical operating rules.

## Metrics and output conventions

The metric file records training N/cycles, held-out N, positive/negative counts,
prevalence, ROC-AUC, average precision (not trapezoidal PR area), Brier score,
log loss, mean probability, probability sample SD (ddof=1), and fit warnings.
Only log loss clips float64 probabilities to machine epsilon and 1-epsilon;
other probability metrics use the original predictions. All probabilities are
checked to be finite and in [0,1].

At each fixed threshold the output includes TP, FP, TN, FN, sensitivity,
specificity, PPV, NPV, balanced accuracy, accuracy, F1, predicted-positive
fraction, FPR, FNR, and Youden J. The existing threshold-metric function is
reused solely to calculate metrics. Undefined PPV/NPV are missing rather than
zero. The summary is long format, with scope, model, optional operating point,
metric, cycle count, valid-cycle count, mean, sample SD, minimum, maximum,
and range. Cycle summaries give equal weight to each held-out cycle, not to
participants; unequal cycle sizes are retained in the detailed outputs.
If any cycle has an undefined rate, that rate's summary remains missing instead
of silently averaging only the other cycles. No sensitivity target-deviation
summary is created for default/Youden.

Ranges and SDs describe instability; no formal clinical acceptability cutoff
or arbitrary binary stability rule is introduced. The four evaluations share
overlapping training data, so their SD is not a standard error or confidence
interval. This analysis is not used to select models or thresholds again.

## Conditional nature and measurement-protocol concern

Model hyperparameters and thresholds were previously chosen using the full
development cohort, including these held-out cycles. Although each new
preprocessing/model fit excludes the test cycle, these earlier choices include
test-cycle information. This is a conditional robustness check of existing
decisions, not an entirely independent assessment of model/threshold selection.

BP harmonization is unchanged: 2011-2016 uses legacy measurements and
2017-2020 uses oscillometric measurements. The 2017-2020 held-out evaluation
therefore trains entirely on legacy-BP cycles and tests on the oscillometric
cycle. For older held-out cycles, training includes the oscillometric cycle.
This asymmetry is relevant, but year, population composition, prevalence,
measurement method, cycle duration, missingness, and training sample size are
confounded. A performance difference cannot be attributed specifically to BP
or used to claim harmonization failure. No BP ablation or harmonization change
is performed here.

2017-2020 has the lowest ROC-AUC and highest Brier/log loss for both models.
XGBoost ROC-AUC is 0.8196 versus 0.8276-0.8362 for legacy held-outs; LR is
0.8037 versus 0.8066-0.8182. Relative to the equal-weight legacy-cycle mean,
the AUC difference is about -0.0110 for XGBoost and -0.0070 for LR. Average
precision is within the legacy-cycle range, not uniformly worse. Diabetes
prevalence is highest at 15.84%; mean predictions near 15.0% understate that
cycle's prevalence by about 0.8 percentage points. Training N is also smallest
when it is held out (16,615 versus roughly 20,000 for other held-outs).
This is a modest cross-cycle signal worth retaining as a protocol-transfer
concern, not evidence of a clinically material effect or causal BP explanation.

XGBoost beats LR on AUC, average precision, Brier, and log loss in every held-out
cycle. Its fixed 80% reference achieves sensitivity 77.70%-81.04%, and its 90%
reference achieves 88.35%-90.70%. LR ranges are 76.28%-82.18% and
87.93%-91.55%. The largest shortfalls from the intended targets occur in
2011-2012 for both models. Thus 2017-2020 is the weaker threshold-free cycle,
while 2011-2012 is the main target-sensitivity undershoot cycle. No cycle shows
a collapse in discrimination, but sensitivity targets do not transport exactly.

Additional limits: only four correlated cycle evaluations, unweighted sample
metrics rather than survey-weighted population inference, prevalence-dependent
AP/PPV/NPV and Brier, and the diagnosed-diabetes target rather than incident
risk or undiagnosed disease. No formal materiality or significance margin is
prespecified. No new tuning, calibration, threshold selection, temporal or
external validation, explanation analysis, subgroup/fairness analysis, final
model selection, or clinical risk bands are performed.

## Execution and tests

```powershell
.venv/Scripts/python.exe -m scripts.analyze_nhanes_cycle_robustness
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_cycle_final_20261017
```

Choose a new unused basetemp if Windows prevents reuse. The existing
test_nhanes.py integration tests are excluded because they access the real
temporal cohort. New tests inspect each model and preprocessing fit's exact
training rows/cycles, confirm disjoint SEQNs and held-out rows, compare fitted
imputation statistics to training-only medians, verify fixed parameters, valid
probabilities, full-precision threshold loading, no calibration/reselection,
output schemas, target deviations, summary arithmetic, and input immutability.
Synthetic sentinel files test frozen-data/threshold hashes without opening
actual temporal data. The production runner checks real development and prior
threshold-file hashes before and after execution.


## Recorded results

Development-only pytest: **39 passed in 32.06 seconds**. No production warnings.
Verified eight model/cycle evaluations, 32 fixed-threshold rows, 100 summary rows,
unchanged development and prior-threshold hashes, and exact reuse of all saved thresholds.

| Held-out cycle | N | Positive | Negative | Prevalence | Training N |
|---|---:|---:|---:|---:|---:|
| 2011_2012 | 5438 | 704 | 4734 | 0.12946 | 20142 |
| 2013_2014 | 5592 | 722 | 4870 | 0.12911 | 19988 |
| 2015_2016 | 5585 | 844 | 4741 | 0.15112 | 19995 |
| 2017_2020 | 8965 | 1420 | 7545 | 0.15839 | 16615 |

| Cycle | Model | ROC-AUC | Average precision | Brier | Log loss | Mean probability | Probability SD |
|---|---|---:|---:|---:|---:|---:|---:|
| 2011_2012 | LR | 0.80714 | 0.38505 | 0.09478 | 0.31266 | 0.13473 | 0.14442 |
| 2011_2012 | XGBoost | 0.82759 | 0.43754 | 0.09075 | 0.29932 | 0.13322 | 0.14414 |
| 2013_2014 | LR | 0.81818 | 0.41641 | 0.09247 | 0.30553 | 0.13829 | 0.14477 |
| 2013_2014 | XGBoost | 0.83622 | 0.44647 | 0.08963 | 0.29431 | 0.13970 | 0.15121 |
| 2015_2016 | LR | 0.80661 | 0.41870 | 0.10668 | 0.34345 | 0.14904 | 0.15700 |
| 2015_2016 | XGBoost | 0.82778 | 0.46303 | 0.10180 | 0.32877 | 0.14831 | 0.15980 |
| 2017_2020 | LR | 0.80367 | 0.42200 | 0.11107 | 0.35482 | 0.15031 | 0.14617 |
| 2017_2020 | XGBoost | 0.81957 | 0.45391 | 0.10764 | 0.34334 | 0.15014 | 0.15170 |

Each cell below is achieved sensitivity / specificity, in percent.

| Cycle | Model | Default 0.50 | Fixed 80% reference | Fixed 90% reference | Fixed Youden reference |
|---|---|---:|---:|---:|---:|
| 2011_2012 | LR | 14.06 / 98.33 | 76.28 / 69.67 | 87.93 / 56.53 | 74.72 / 70.95 |
| 2011_2012 | XGBoost | 15.06 / 98.77 | 77.70 / 70.95 | 88.35 / 59.13 | 79.97 / 68.50 |
| 2013_2014 | LR | 14.96 / 98.46 | 79.36 / 69.32 | 89.20 / 54.48 | 78.12 / 70.78 |
| 2013_2014 | XGBoost | 19.39 / 98.32 | 80.75 / 70.99 | 89.20 / 57.45 | 82.13 / 68.60 |
| 2015_2016 | LR | 16.82 / 97.53 | 80.33 / 67.92 | 89.22 / 53.95 | 77.73 / 69.50 |
| 2015_2016 | XGBoost | 19.55 / 97.62 | 81.04 / 69.46 | 90.52 / 56.04 | 83.18 / 67.31 |
| 2017_2020 | LR | 13.80 / 98.28 | 82.18 / 64.49 | 91.55 / 50.18 | 80.28 / 66.64 |
| 2017_2020 | XGBoost | 16.20 / 97.83 | 79.58 / 68.03 | 90.70 / 53.47 | 82.96 / 64.65 |

Cross-cycle transportability (ranges and sample SDs; deviations in percentage points):

| Model | Reference | Sensitivity range | Sensitivity SD | Specificity range | Specificity SD | Mean absolute target deviation | Max absolute target deviation |
|---|---|---:|---:|---:|---:|---:|---:|
| LR | 80% | 76.28-82.18% | 2.47 pp | 64.49-69.67% | 2.36 pp | 1.72 pp | 3.72 pp |
| LR | 90% | 87.93-91.55% | 1.51 pp | 50.18-56.53% | 2.65 pp | 1.30 pp | 2.07 pp |
| XGBoost | 80% | 77.70-81.04% | 1.52 pp | 68.03-70.99% | 1.41 pp | 1.13 pp | 2.30 pp |
| XGBoost | 90% | 88.35-90.70% | 1.12 pp | 53.47-59.13% | 2.40 pp | 0.92 pp | 1.65 pp |

Development SHA-256: `f75e4db96bc672ea0f0f6b1913c484d4af56a9f0386541fca07cd22b5e4e15d9`.

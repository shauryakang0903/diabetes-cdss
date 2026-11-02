# Development-only calibration comparison

## Scope and fixed models

Only `data/processed/nhanes/nhanes_development_2011_2020.csv` is read. The shared
loader validates adult ages, binary outcomes, unique IDs, development cycle
labels, and the frozen counts: 25,580 adults, 3,690 positives. The actual
2021â€“2023 temporal cohort is not accessed, even for a hash. Development bytes
are hashed before and after the experiment. Previous results and frozen CSVs
are not overwritten.

Core-10 and preprocessing are unchanged: age, sex, race_ethnicity, bmi,
waist_cm, mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl,
total_cholesterol_mg_dl, smoking_status. The seven numeric features use median
imputation, with StandardScaler for LR only. Sex, race/ethnicity, and smoking
status use most-frequent imputation and OneHotEncoder(handle_unknown="ignore").
Every preprocessing fit is inside the full sklearn pipeline.

LR: C=0.1, L2 (l1_ratio=0 in installed sklearn 1.9), lbfgs, max_iter=2000,
class_weight=None, seed 42. Explicit deprecated penalty arguments are avoided.
XGBoost: 200 trees, learning_rate=0.03, max_depth=4, min_child_weight=10,
subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=10,
binary:logistic objective, scale_pos_weight=1, hist trees on CPU, gamma=0,
eval_metric=logloss, n_jobs=1, seed 42. There is no early stopping, sample
weighting, class weighting, or hyperparameter search. Full estimator parameters
and package versions are recorded in the provenance JSON. XGBoost's missing
value NaN sentinel is encoded as the string "NaN" for strict JSON serialization.
No new packages were needed; the baseline dependency pins remain applicable.

## Leakage-safe calibration design

All six candidates use identical outer StratifiedKFold splits: five folds,
shuffle=True, random_state=42. Each outer training set has 20,464 rows and each
outer test set has 5,116 rows. Fresh models are used for every candidate/fold.

Raw models fit the full pipeline on the outer-training rows. Sigmoid and
isotonic candidates wrap the full pipeline in CalibratedClassifierCV with
ensemble=False, n_jobs=1, and three shuffled stratified inner folds, seed 42.
Inner indices are generated relative to the current outer-training set only
and shared across both calibration methods and models.

Within each calibrated fit, sklearn clones and fits the full base pipeline on
each inner-training subset, then predicts scores for its disjoint inner-held-out
subset. Every outer-training participant contributes exactly one out-of-fold
score to calibrator training. Sklearn then refits a fresh base pipeline on all
outer-training rows and learns a single calibrator from the assembled OOF
scores and labels. Only this base-model/calibrator pair predicts the untouched
outer test fold. There is no prefit/frozen wrapper and no calibration on the
outer test labels. The calibration fit therefore sees scores from models trained
on about two-thirds of outer training, while the final base model uses all of
outer training; this training-size distribution shift is a limitation of the
chosen ensemble=False design. It avoids confounding calibration with prediction
averaging from an ensemble, and keeps raw/calibrated base fits comparable.

The installed sklearn implementation prefers decision_function if available:
LR calibration uses its decision scores; XGBoost calibration uses its positive
class probabilities because it lacks decision_function. In particular, sigmoid
on XGBoost probabilities is not a logistic recalibration of their logits. This
is the standard wrapper behavior, kept fixed rather than modified after results.
Isotonic maps scores monotonically with flat regions, which can create ranking
ties. Each calibrated method is evaluated only on the outer-held-out cohort.
No full-development final fit or final model selection occurs.

These fold boundaries protect against calibration/preprocessing leakage.
However, provisional hyperparameters and preprocessing were previously selected
using this development cohort. This is a conditional comparison of prespecified
candidates, not a fully nested re-estimation of all prior research decisions.
It must not be described as an untouched temporal evaluation or a completely
unbiased estimate after all stages of model selection.

## Probability diagnostics and conventions

- ROC-AUC and average precision use outer-held-out probabilities. Average
  precision is the PR summary, not trapezoidal PR-AUC.
- Brier score uses unaltered probabilities. Log loss uses float64 probabilities
  clipped to machine epsilon (2.220446049250313e-16) and one minus epsilon,
  consistent with standard float64 sklearn log loss. This preserves the severe
  penalty for confidently wrong exact zero/one predictions.
- Joint calibration line: fit an unpenalized logistic regression by minimizing
  mean binary negative log likelihood, using
  `logit P(Y=1) = intercept + slope * logit(p)`. Ideal values are intercept=0,
  slope=1. Both coefficients are estimated together. This intercept is **not**
  the calibration-in-the-large intercept obtained with slope fixed at one.
  Only for this diagnostic, clip p to [1e-6, 1-1e-6] before logits. Optimization
  uses scipy BFGS, analytic gradients, maxiter=2000, gtol=1e-8. Constant scores,
  unsuccessful optimization, nonfinite coefficients, or ill-conditioned Hessian
  (>1e12 condition number) produce warnings and missing coefficients. These
  diagnostic regressions use outer-test labels to measure calibration; they
  never change evaluated predictions or fit a deployed calibrator.
- ECE uses ten prespecified equal-width bins from zero to one, left-inclusive
  and right-exclusive, with the last bin also containing p=1. It is
  `sum(n_bin / N * abs(mean_prediction_bin - observed_rate_bin))`. Empty bins
  have zero count and missing means. ECE is bin-dependent and supportive only.
- Accuracy, sensitivity, specificity, precision, and F1 use the descriptive
  threshold 0.5. No clinical threshold or risk bands are selected. Undefined
  precision/F1 use zero_division=0. Specificity is TN/(TN+FP).

Summary results are the mean and sample SD (ddof=1) of five outer-fold metrics.
If any diagnostic coefficient is missing, its summary remains missing rather
than silently averaging fewer folds. Overlapping training samples make fold
scores dependent: SD is not a confidence interval or a standard error.

The curve file contains ten rows for each candidate/outer fold (scope
outer_fold) plus ten pooled outer-OOF points per candidate (scope
pooled_outer_oof, outer_fold=0). Pooled curves concatenate untouched outer-fold
predictions for descriptive plotting; they are not used to fit calibrators.
Their ECE need not equal the mean fold ECE. Every point records bin edges,
sample count, mean probability, and observed outcome rate. Sparse tail bins
should not be interpreted as stable estimates.

## Execution and tests

```powershell
.venv/Scripts/python.exe -m scripts.compare_nhanes_calibration
.venv/Scripts/python.exe -m pytest -q --ignore=tests/test_nhanes.py -p no:cacheprovider --basetemp=.test_tmp_calibration_final_20261017
```

Choose a new unused basetemp if Windows prevents reuse. The excluded existing
data-engineering integration tests read real temporal-validation data; all
other project tests are run. New tests construct all six candidates, fit real
calibration wrappers on small synthetic data with missing values, inspect
every base fit and OOF scoring row set, verify no fit/prediction overlap and
identical outer folds, validate metric arithmetic and ideal calibration-line
conventions, handle zero/one probabilities, check reproducibility and output
schemas, reject temporal labels, restrict file reads to the development input,
and check synthetic frozen-file hashes. Production separately checks the
actual development-file hash. A deliberately constant small-sample isotonic
test case checks warnings and missing diagnostic values.

## Interpretation and next-stage recommendation

No calibration method demonstrates a material probability-quality improvement
over its raw model in this run. LR sigmoid is effectively unchanged; LR isotonic
has a tiny Brier reduction but worse log loss and average precision. XGBoost
sigmoid has worse Brier score, log loss, and ECE despite a joint calibration
line closer to ideal. XGBoost isotonic has near-ideal line coefficients and
reasonable central pooled-bin behavior, but slightly worse Brier/log loss and
lower average precision than raw XGBoost. Joint line coefficients alone are
insufficient evidence of reliable probabilities across the score range.

Sigmoid preserves discrimination here. Isotonic introduces ties and reduces
average precision; its ROC-AUC changes are small. There is no demonstrated
clinical materiality or significance threshold for these differences.
No production fit/diagnostic warnings were recorded. Isotonic nevertheless
produced exact 0/1 predictions: LR had 17 zeros and 4 ones across outer OOF
predictions; XGBoost had 135 zeros and 2 ones. These extremes and sparse upper
bins are reasons for caution, including log-loss sensitivity to confident errors.

No calibrated variant earns preference for threshold analysis on these results.
Carry raw XGBoost forward as the provisional predictive reference and raw LR
as the interpretable benchmark. If a calibrated sensitivity comparator is
desired in the next stage, XGBoost-isotonic is the defensible optional arm,
not a replacement established by this experiment. XGBoost-sigmoid should not
be preferred simply because its slope/intercept are closer to ideal. These
are recommendations for a later experiment, not a final model selection.

Other limits remain: unweighted sample-level random CV mixes development
cycles and does not establish population or temporal generalization; target
is physician-diagnosed diabetes, not prospective incident risk or undiagnosed
disease detection. No threshold optimization, temporal validation, subgroup
analysis, explanation analysis, or risk-band definition was performed.


## Recorded results

Development-only pytest: **28 passed in 30.37 seconds**. No production warnings.

| Candidate | roc_auc | average_precision | brier_score | log_loss |
|---|---:|---:|---:|---:|
| LR raw | 0.80960 ± 0.00952 | 0.41588 ± 0.01221 | 0.10236 ± 0.00163 | 0.33196 ± 0.00553 |
| LR sigmoid | 0.80960 ± 0.00952 | 0.41588 ± 0.01221 | 0.10236 ± 0.00163 | 0.33196 ± 0.00553 |
| LR isotonic | 0.80893 ± 0.00893 | 0.40323 ± 0.01142 | 0.10235 ± 0.00132 | 0.33427 ± 0.00650 |
| XGB raw | 0.82658 ± 0.00933 | 0.45200 ± 0.01594 | 0.09884 ± 0.00171 | 0.32022 ± 0.00574 |
| XGB sigmoid | 0.82658 ± 0.00933 | 0.45200 ± 0.01594 | 0.10050 ± 0.00201 | 0.33019 ± 0.00590 |
| XGB isotonic | 0.82607 ± 0.00894 | 0.44256 ± 0.01693 | 0.09890 ± 0.00179 | 0.32154 ± 0.00811 |

| Candidate | calibration_intercept | calibration_slope | ece |
|---|---:|---:|---:|
| LR raw | 0.00793 ± 0.09266 | 1.00556 ± 0.05840 | 0.01080 ± 0.00107 |
| LR sigmoid | 0.00722 ± 0.09282 | 1.00533 ± 0.05820 | 0.01084 ± 0.00121 |
| LR isotonic | -0.02877 ± 0.08079 | 0.97784 ± 0.05147 | 0.00766 ± 0.00273 |
| XGB raw | 0.08235 ± 0.09618 | 1.06038 ± 0.06439 | 0.01060 ± 0.00342 |
| XGB sigmoid | 0.01743 ± 0.10739 | 1.01142 ± 0.05941 | 0.03020 ± 0.00405 |
| XGB isotonic | -0.01923 ± 0.09574 | 0.98634 ± 0.06511 | 0.01107 ± 0.00225 |

| Candidate | accuracy | sensitivity | specificity | precision | f1 |
|---|---:|---:|---:|---:|---:|
| LR raw | 0.86145 ± 0.00134 | 0.14580 ± 0.01180 | 0.98209 ± 0.00122 | 0.57822 ± 0.01658 | 0.23274 ± 0.01588 |
| LR sigmoid | 0.86149 ± 0.00127 | 0.14607 ± 0.01187 | 0.98209 ± 0.00127 | 0.57871 ± 0.01568 | 0.23312 ± 0.01581 |
| LR isotonic | 0.86165 ± 0.00134 | 0.13686 ± 0.01687 | 0.98383 ± 0.00257 | 0.58894 ± 0.02440 | 0.22161 ± 0.02198 |
| XGB raw | 0.86591 ± 0.00282 | 0.17805 ± 0.01645 | 0.98186 ± 0.00221 | 0.62340 ± 0.03490 | 0.27676 ± 0.02181 |
| XGB sigmoid | 0.86474 ± 0.00290 | 0.22466 ± 0.01809 | 0.97264 ± 0.00419 | 0.58183 ± 0.02918 | 0.32366 ± 0.01881 |
| XGB isotonic | 0.86536 ± 0.00289 | 0.18509 ± 0.01956 | 0.98004 ± 0.00401 | 0.61222 ± 0.04113 | 0.28358 ± 0.02260 |

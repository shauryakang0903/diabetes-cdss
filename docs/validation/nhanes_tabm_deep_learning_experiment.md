# Secondary development-only TabM experiment

## Question and scope

Can a modern tabular deep-learning architecture improve discrimination or probability quality relative to tuned trees and logistic regression on the same compact clinical inputs? Deep learning is not assumed to be superior. This is an experiment on physician-diagnosed diabetes status derived from DIQ010, not future incident diabetes. Existing XGBoost remains the temporally evaluated primary CDSS model. There is no clinical deployment claim.

The 2021–2023 cohort and its result files are not loaded for this experiment. That cohort had already been used in prior validation and was intentionally excluded. TabM results are development-only. No changes are made to the primary artifact, metadata, app, thresholds, recommendations, input-quality logic, SHAP, subgroup, DCA or temporal results. Integrity hashing of saved files does not parse or use their results.

## Architecture and environment

The official [Yandex Research TabM package](https://github.com/yandex-research/tabm) implements the architecture from [Gorishniy et al., ICLR 2025](https://arxiv.org/abs/2410.24210). The installed `TabM.make` source and official training guidance were inspected. It emits one logit per member; training minimizes mean independent member binary cross-entropy, while inference averages sigmoid probabilities, not logits. This uses `arch_type="tabm"`; no substitution with an MLP, TabNet or transformer.

Environment: Python 3.13.4, TabM 0.0.3, PyTorch 2.15.0+cpu, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, rtdl_num_embeddings 0.0.12. Neither torch nor tabm was previously installed. Pip installed only required dependencies; `pip check` passed. Versions are pinned in `requirements-nhanes-tabm.txt` and recorded in the summary.

The host reports an older 2 GB NVIDIA GPU with driver 511.69/CUDA 11.6. The compatible project package installation is CPU-only (`torch.cuda.is_available() == False`, `torch.version.cuda is None`). No system driver or Python environment was replaced. Runs use two CPU threads. The runner chooses CUDA automatically in a compatible environment, with deterministic algorithms enabled, and fails on unsupported deterministic operations rather than silently accepting nondeterminism.

## Inputs and preprocessing

Exactly the original Core-10 features are used: age, sex, race_ethnicity, bmi, waist_cm, mean_systolic_bp, mean_diastolic_bp, hdl_mg_dl, total_cholesterol_mg_dl, smoking_status. No laboratory target proxies, cycle predictor or additional feature is included.

The frozen 2011–March 2020 development cohort has 25,580 adults: 3,690 positive and 21,890 negative. No participant is removed because a predictor is missing.

Numeric inputs use median imputation and standardization. Categorical inputs use mode imputation and a training-only integer mapping, passed to TabM's native categorical interface (which internally one-hot encodes). Code zero reserves unknown categories; observed categories map to 1..N, so unseen assessment categories do not cause indexing failures. Imputers retain all-empty training columns. Every gradient-training fit gets fresh preprocessing; stopping and scoring rows do not fit any statistics. The all-outer-training refit fits its own fresh preprocessing. There is no target encoding, weighting, calibration or threshold optimization.

## Nested design and compute budget

Outer StratifiedKFold: five folds, shuffle=True, seed 42, preserving original development-file row order. This reproduces the earlier nested-CV fold-generation code. Participant identifiers, outcomes and fold assignments are verified against saved classical OOF predictions before pairing.

Inner StratifiedKFold: three folds, shuffle=True, seed 42. For each inner-training set, stratified 15% is reserved exclusively for early stopping. Preprocessing and gradient updates use the remaining 85%; the separate inner-assessment fold supplies the ROC-AUC for configuration selection. Candidate score is mean inner-assessment ROC-AUC; exact ties select the first predeclared candidate.

After selecting a configuration within each outer fold, a new 85%/15% outer-training partition determines its epoch count. A fresh model and preprocessing are then fitted on **all** outer-training rows for exactly that epoch count. The outer-test fold is scored once and is never used for early stopping or selection. This yields one OOF probability for each development participant. Split audits record row-index hashes and counts; the complete assignments are reproducible from the documented seeds and saved OOF fold labels.

Early stopping maximizes stopping ROC-AUC, with patience 20 and cap 200 epochs; earliest epoch wins an exact tie. Seeds for Python, NumPy and torch (CPU and CUDA, when usable) are 42 + 100*outer_fold + inner_fold; the outer epoch-selection/refit uses +90 instead of inner_fold. Split seeds follow the same formula. Outer and inner CV seeds remain 42. A single training seed sequence is used; repeated-seed uncertainty is not estimated.

Official defaults were inspected: three blocks, width 512, dropout 0.1, k=32, ReLU; official AdamW guidance specifies learning rate 0.002 and weight decay 0.0003. A three-epoch official-default smoke fit was interrupted because CPU cost was prohibitive, without inspecting any outer-test results. The primary predeclared CPU search preserves full nested CV while reducing capacity to two configurations:

| Parameter | Candidate 0 | Candidate 1 |
|---|---:|---:|
| n_blocks | 2 | 3 |
| d_block | 128 | 128 |
| dropout | 0.1 | 0.2 |
| k | 8 | 8 |
| learning_rate | 0.002 | 0.002 |
| weight_decay | 0.0003 | 0.0003 |
| batch_size | 512 | 512 |

This is 30 inner fits, five outer epoch-selection fits and five fixed-epoch outer refits, not an exhaustive search. No numerical feature embeddings are used. These compromises limit conclusions about larger TabM variants or TabM with embeddings; a negative result cannot establish that all deep learning is inferior. A successful three-epoch compact-model smoke run precedes the primary experiment.

## Metrics and classical comparison

Each outer fold records ROC-AUC, average precision (not trapezoidal PR area), Brier, binary log loss, and accuracy/sensitivity/specificity/precision/F1 at 0.5. Summary means and sample SD (ddof=1) describe five folds, not five independent patient cohorts. Metrics are also saved at 0.14026810228824615: this threshold was developed for XGBoost and is only a descriptive cross-model reference, not an optimized or clinical TabM cutoff.

Primary LR/XGBoost comparison loads `nested_tuning_outer_folds.csv`; it does not hardcode published approximate scores. XGBoost selections were identical across all earlier outer folds. Saved `threshold_oof_predictions.csv` is usable for XGBoost pairing only after its participant identities, targets, outer folds and fold metrics reproduce the archived nested results. The runner fails rather than substituting a mismatched comparator.

The nested LR experiment selected C=1 in fold 3, while the saved threshold LR OOF used C=0.1 everywhere. To recover exact nested LR log loss, the reporting step replays only that mismatched fold using its archived selected C, original training rows and fresh original preprocessing. It performs no search. Its metrics must reproduce the archived nested fold to numerical precision before its probabilities are accepted (the measured maximum difference was 1.11e-16). Other folds reuse matching saved OOF predictions. The recovered probabilities and audit are saved separately under deep_learning; original LR results are never overwritten, and XGBoost is never loaded or refitted. Other LR metrics still come directly from the original nested artifact.

Paired TabM-minus-XGBoost differences use 1,000 IID participant bootstrap replicates with seed 42, percentile 95% intervals, for ROC-AUC, AP and Brier. The observed pooled-OOF difference and bootstrap mean difference are both reported. This resamples aligned predictions without refitting models. Intervals are conditional on those fitted folds and do not capture all training, search or repeated-CV uncertainty; NHANES survey clustering and weights are not incorporated. Fold-level SD and bootstrap intervals must not be treated as independent-replication evidence.

## Outputs and rerun

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts/run_nhanes_tabm_experiment.py --smoke --compact-search --threads 2
.venv\Scripts\python.exe scripts/run_nhanes_tabm_experiment.py --compact-search --max-epochs 200 --patience 20 --threads 2
```

Results are separate under `results/nhanes/deep_learning/`: fold metrics, summary/provenance JSON, OOF predictions, inner search, split audit, training histories, comparison, bootstrap and matplotlib figures. No final experimental checkpoint is required or saved; this avoids choosing a single final configuration after observing outer performance. The checkpoint-metadata helper and summary explicitly state experimental/development-only/not-primary status. No TabM integration into app.py is made. Exploratory explainability is omitted to retain focus on predictive comparison.

## Scientific limitations

The cohort and previous LR/XGBoost results are familiar from prior project development, so this is a secondary research comparison rather than a prospectively untouched benchmark. TabM's smaller search has a different computational budget from the earlier tree search. Single-seed compact architectures, no numeric embeddings and CPU constraints limit generality. Random pooled-cycle CV does not demonstrate temporal transportability. Missingness is imputed, not eliminated; the target is diagnosis history rather than biological undiagnosed disease. Fold-wise probability quality is measured without recalibration. Survey design and within-household clustering are not modeled. No clinical operating point or production-model replacement is selected.

Completed results and measured runtime follow below.

Validation uses the dedicated `tests/test_nhanes_tabm.py` suite. The historical full suite is intentionally not invoked here because it contains temporal-cohort loading/evaluation tests; this task forbids that access. Deterministic seeds do not guarantee bitwise equality across different devices, thread counts or package versions.

## Completed results (2026-11-01)

All entries below are development-only. Means and sample SD summarize five outer folds.

| Model | ROC-AUC | AP | Brier | Log loss | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.809621 ± 0.009491 | 0.415944 ± 0.012120 | 0.102361 ± 0.001635 | 0.331957 ± 0.005524 | 0.861454 ± 0.001336 | 0.146612 ± 0.011750 | 0.981955 ± 0.001361 | 0.577820 ± 0.017030 | 0.233733 ± 0.015738 |
| XGBoost | 0.826579 ± 0.009334 | 0.451996 ± 0.015945 | 0.098840 ± 0.001710 | 0.320216 ± 0.005742 | 0.865911 ± 0.002822 | 0.178049 ± 0.016445 | 0.981864 ± 0.002212 | 0.623401 ± 0.034899 | 0.276762 ± 0.021814 |
| TabM | 0.828624 ± 0.006663 | 0.455531 ± 0.014298 | 0.098765 ± 0.001414 | 0.319036 ± 0.004617 | 0.865715 ± 0.002292 | 0.188347 ± 0.043645 | 0.979899 ± 0.007275 | 0.618559 ± 0.036125 | 0.285676 ± 0.047731 |

TabM minus XGBoost mean-fold differences: roc_auc +0.002045, average_precision +0.003535, brier_score -0.000076, log_loss -0.001180.

### Paired fixed-OOF bootstrap

These pooled-OOF differences are not identical to mean-fold differences for nonlinear metrics. Negative Brier differences favor TabM.

| Metric | Observed pooled-OOF difference | Bootstrap mean difference | Percentile 95% interval |
|---|---:|---:|---|
| roc_auc | +0.001234 | +0.001197 | [-0.000853, +0.003206] |
| average_precision | +0.000954 | +0.000994 | [-0.005004, +0.006842] |
| brier_score | -0.000076 | -0.000072 | [-0.000503, +0.000379] |

Each metric used 1,000 participant bootstrap replicates, seed 42. All intervals include zero. The small numerical mean improvements do not establish a clear TabM advantage over XGBoost. Probability quality is very similar. TabM improves descriptively over LR, but this is not independent temporal or clinical validation.

### Selected configurations

All fits used width 128, k=8, AdamW learning rate 0.002, weight decay 0.0003, batch size 512, ReLU and no numerical embeddings.

| Outer fold | Blocks | Dropout | Selected refit epochs |
|---|---:|---:|---:|
| 1 | 3 | 0.2 | 52 |
| 2 | 3 | 0.2 | 30 |
| 3 | 2 | 0.1 | 17 |
| 4 | 2 | 0.1 | 51 |
| 5 | 3 | 0.2 | 45 |

Three blocks/dropout 0.2 were selected in folds 1, 2 and 5; two blocks/dropout 0.1 in folds 3 and 4. No global final configuration was chosen.

### Descriptive XGBoost threshold on TabM

At 0.14026810228824615, TabM mean ± SD sensitivity is 0.808943 ± 0.028568; specificity 0.684376 ± 0.025859; precision 0.302294 ± 0.012029; accuracy 0.702346 ± 0.018592; F1 0.439783 ± 0.009871. This is not an optimized TabM or clinical threshold.

### Cost, integrity and interpretation

Primary nested execution took 47.03 minutes on two CPU threads; 40 TabM fits and 2004 total training epochs across fits. Fit time summed to 46.22 minutes. No inner or outer early-stopping fit reached the 200-epoch cap. The interrupted default-model probe and dependency installation are excluded from this primary runtime.

OOF coverage: 25,580 unique participants, one probability each, 3,690 positive outcomes, five folds of 5,116. Probabilities range from 0.000369852 to 0.889242828. All 98 protected files remained byte-identical. The temporal cohort was never opened; temporal result values were not used.

Frozen XGBoost artifact SHA-256, before = after: fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528.

Frozen metadata SHA-256, before = after: 57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3.

No experimental weights were saved. Existing XGBoost remains the primary CDSS model; app.py and all decision-support logic are unchanged. Results demonstrate comparable compact-TabM and boosted-tree development performance, not evidence that deep learning is inherently better or worse.

The final reporting revision adds LR recovery and preflight integrity checks only; numerical TabM training is unchanged. Original training-run and final reporting-source hashes are retained separately.

Targeted verification: 11 tests passed, no failures. Coverage includes train-only fitting, early-stopping/inner/outer separation, categorical unknowns and missingness, deterministic probabilities, OOF integrity, metrics, LR recovery without XGBoost fitting, frozen SHA checks, and preflight failure before training when protected inputs change. The final real-data reporting step also passed its OOF and protected-file assertions.

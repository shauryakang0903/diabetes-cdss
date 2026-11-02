"""First frozen-development temporal evaluation; evaluation never updates models."""

import hashlib
import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data.load_nhanes import validate_cohort_separation
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_baseline_comparison import MODELS
from src.evaluation.nhanes_calibration_comparison import (
    BIN_EDGES, LOGIT_EPSILON, evaluate_probabilities, make_provisional_pipeline,
)
from src.evaluation.nhanes_cycle_robustness import load_fixed_thresholds
from src.evaluation.nhanes_preprocessing_comparison import DEVELOPMENT_PATH, OUTPUT_DIR
from src.evaluation.nhanes_threshold_analysis import threshold_metrics

VALIDATION_PATH = DEVELOPMENT_PATH.parent / "nhanes_validation_2021_2023.csv"
REFERENCE_PATHS = {
    "thresholds": OUTPUT_DIR / "threshold_candidate_points.csv",
    "nested": OUTPUT_DIR / "nested_tuning_summary.csv",
    "calibration": OUTPUT_DIR / "calibration_comparison_summary.csv",
    "cycles": OUTPUT_DIR / "cycle_robustness_thresholds.csv",
}
RATE_COMPARISONS = ("sensitivity", "specificity", "precision", "negative_predictive_value",
                    "accuracy", "balanced_accuracy", "f1", "predicted_positive_fraction")


def audit_cohorts(development, validation):
    """Read-only checks/description before any model fit; no adapted preprocessing."""
    validate_cohort_separation(development, validation)
    audit = dict(seqn_overlap_n=0, features=CORE_FEATURES, cohorts={})
    for name, frame in [("development", development), ("temporal_validation", validation)]:
        if set(frame.diabetes.unique()) != {0, 1}:
            raise ValueError(f"{name}: both outcome classes are required")
        audit["cohorts"][name] = dict(n=len(frame), positive_n=int(frame.diabetes.sum()),
            negative_n=int(frame.diabetes.eq(0).sum()), prevalence=float(frame.diabetes.mean()),
            feature_missingness={c: dict(n=int(frame[c].isna().sum()),
                                         fraction=float(frame[c].isna().mean())) for c in CORE_FEATURES})
    n_missing = validation[CORE_FEATURES].isna().sum(axis=1)
    audit["validation_missing_field_counts"] = {
        "0": int(n_missing.eq(0).sum()), "1": int(n_missing.eq(1).sum()),
        "2": int(n_missing.eq(2).sum()), "3+": int(n_missing.ge(3).sum())}
    return audit


def load_cohorts():
    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    audit = audit_cohorts(development, validation)
    for name, n, positive in [("development", 25580, 3690), ("temporal_validation", 7536, 1070)]:
        actual = audit["cohorts"][name]
        if actual["n"] != n or actual["positive_n"] != positive:
            raise ValueError(f"{name}: frozen cohort counts do not match specification")
    return development, validation, audit


def load_references():
    fixed = load_fixed_thresholds(REFERENCE_PATHS["thresholds"])
    frames = {key: pd.read_csv(path, float_precision="round_trip") for key, path in REFERENCE_PATHS.items()}
    return fixed, frames


def completeness_metrics(validation, probability, model):
    """Stratify already-generated probabilities; never refit or re-predict groups."""
    complete = validation[CORE_FEATURES].notna().all(axis=1).to_numpy()
    rows = []
    for label, mask in [("all_core10_observed", complete), ("one_or_more_core10_missing", ~complete)]:
        y, p = validation.diabetes.to_numpy()[mask], np.asarray(probability, dtype=float)[mask]
        n = len(y)
        both = n > 0 and len(np.unique(y)) == 2
        rows.append(dict(model=model, completeness_group=label, n=n,
            positive_n=int(y.sum()), negative_n=int(np.sum(y == 0)),
            prevalence=float(y.mean()) if n else np.nan,
            roc_auc=roc_auc_score(y, p) if both else np.nan,
            # AP is undefined with no positive examples; with all positives it is 1.
            average_precision=average_precision_score(y, p) if n and y.sum() else np.nan,
            brier_score=brier_score_loss(y, p) if n else np.nan,
            outcome_status="both_classes" if both else ("one_class" if n else "empty")))
    return pd.DataFrame(rows)


def add_development_comparisons(row, model, references):
    nested = references["nested"].set_index("model").loc[model]
    raw = references["calibration"]
    raw = raw[(raw.model == model) & (raw.calibration == "raw")]
    if len(raw) != 1:
        raise ValueError(f"Expected one prior raw calibration reference for {model}")
    raw = raw.iloc[0]
    for metric in ("roc_auc", "average_precision", "brier_score"):
        value = float(nested[metric + "_mean"])
        row["development_nested_" + metric] = value
        row["difference_vs_nested_" + metric] = row[metric] - value
        row["absolute_difference_vs_nested_" + metric] = abs(row[metric] - value)
    for metric in ("roc_auc", "average_precision", "brier_score", "log_loss",
                   "calibration_intercept", "calibration_slope", "ece"):
        value = float(raw[metric + "_mean"])
        row["development_raw_oof_mean_" + metric] = value
        row["difference_vs_raw_oof_" + metric] = row[metric] - value


def add_threshold_comparisons(scores, model, references):
    prior = references["thresholds"]
    cycles = references["cycles"]
    rows = []
    for _, record in scores.iterrows():
        row = record.to_dict()
        point = row["operating_point"]
        dev = prior[(prior.model == model) & (prior.operating_point == point)]
        cycle = cycles[(cycles.model == model) & (cycles.operating_point == point)]
        if len(dev) != 1 or len(cycle) != 4 or cycle.held_out_cycle.nunique() != 4:
            raise ValueError("Expected one pooled OOF and four cycle references per operating point")
        for metric in RATE_COMPARISONS:
            value = float(dev.iloc[0][metric])
            row["development_oof_" + metric] = value
            row["difference_vs_development_oof_" + metric] = row[metric] - value
        row["sensitivity_target_deviation"] = row["sensitivity"] - row["target_sensitivity"]
        row["absolute_sensitivity_target_deviation"] = abs(row["sensitivity_target_deviation"])
        for metric in ("sensitivity", "specificity"):
            low, high = float(cycle[metric].min()), float(cycle[metric].max())
            row["cycle_" + metric + "_min"] = low
            row["cycle_" + metric + "_max"] = high
            row["difference_vs_cycle_" + metric + "_min"] = row[metric] - low
            row["difference_vs_cycle_" + metric + "_max"] = row[metric] - high
            row["distance_outside_cycle_" + metric + "_range"] = max(low - row[metric], row[metric] - high, 0)
            row["cycle_" + metric + "_range_status"] = "below" if row[metric] < low else ("above" if row[metric] > high else "within")
        for count in ("tp", "fp", "tn", "fn"):
            row[count] = int(row[count])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_temporal(development, validation, fixed_thresholds, references, output_dir=OUTPUT_DIR):
    audit_cohorts(development, validation)  # Defensive check for callers bypassing the runner.
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError("Cannot write evaluation outputs into frozen processed data")
    metrics_rows, threshold_frames, curves, groups = [], [], [], []
    y = validation.diabetes.astype(int)
    for model in MODELS:
        print(f"Fit ONCE on full development and evaluate frozen {model}", flush=True)
        pipeline = make_provisional_pipeline(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(development[CORE_FEATURES], development.diabetes.astype(int))
            probability = np.asarray(pipeline.predict_proba(validation[CORE_FEATURES])[:, 1], dtype=float)
            metrics, points = evaluate_probabilities(y, probability)
        for warning in caught:
            warnings.warn(f"{model}: {warning.message}", warning.category, stacklevel=2)
        row = dict(model=model, development_n=len(development), temporal_n=len(validation),
            temporal_positive_n=int(y.sum()), temporal_negative_n=int(y.eq(0).sum()),
            observed_prevalence=float(y.mean()), mean_predicted_probability=float(np.mean(probability)),
            fit_count=1, fit_warnings=json.dumps([str(w.message) for w in caught]), **metrics)
        add_development_comparisons(row, model, references)
        metrics_rows.append(row)
        fixed = fixed_thresholds[fixed_thresholds.model == model].reset_index(drop=True)
        scores = threshold_metrics(y, probability, fixed.threshold.to_numpy())
        scores["operating_point"], scores["target_sensitivity"] = fixed.operating_point, fixed.target_sensitivity
        scores["model"] = model
        threshold_frames.append(add_threshold_comparisons(scores, model, references))
        curves.append(points.assign(model=model, cohort="temporal_validation"))
        groups.append(completeness_metrics(validation, probability, model))
    metrics, thresholds = pd.DataFrame(metrics_rows), pd.concat(threshold_frames, ignore_index=True)
    curve, missingness = pd.concat(curves, ignore_index=True), pd.concat(groups, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("metrics", metrics), ("thresholds", thresholds),
                        ("calibration_curve", curve), ("missingness_robustness", missingness)]:
        frame.to_csv(output_dir / f"temporal_validation_{name}.csv", index=False)
    return metrics, thresholds, curve, missingness


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_temporal_validation():
    # Preflight is persisted BEFORE fitting; input and prior-result hashes protect
    # independence/integrity. Subsequent diagnostics never change fitted models.
    inputs = {"development": DEVELOPMENT_PATH, "temporal_validation": VALIDATION_PATH, **REFERENCE_PATHS}
    before = {name: file_hash(path) for name, path in inputs.items()}
    prior_paths = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and not p.name.startswith("temporal_validation_")]
    prior_hashes = {str(p): file_hash(p) for p in prior_paths}
    development, validation, audit = load_cohorts()
    fixed, references = load_references()
    provenance = dict(status="preflight_complete", started_utc=datetime.now(timezone.utc).isoformat(),
        input_paths={name: str(path) for name, path in inputs.items()}, sha256_before=before,
        prior_result_sha256_before=prior_hashes, preflight=audit,
        features=CORE_FEATURES, class_weight="none", calibration="none", tuning=False,
        threshold_reselection=False, random_seed=42, fits_per_model=1,
        fixed_thresholds=[{k: None if pd.isna(v) else v for k, v in record.items()}
                          for record in fixed.to_dict(orient="records")],
        model_parameters={m: {k: "NaN" if isinstance(v, float) and np.isnan(v) else v
            for k, v in make_provisional_pipeline(m).named_steps["model"].get_params().items()} for m in MODELS},
        package_versions={n: version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost")},
        calibration_line="Joint unpenalized logit P(Y=1)=intercept+slope*logit(clipped p); diagnostic only",
        logit_epsilon=LOGIT_EPSILON, ece_bin_edges=BIN_EDGES.tolist(),
        log_loss_epsilon=float(np.finfo(np.float64).eps), probability_threshold_rule="p >= saved threshold",
        nonfinite_parameter_encoding="NaN model-parameter sentinel encoded as string NaN")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance_path = OUTPUT_DIR / "temporal_validation_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    results = evaluate_temporal(development, validation, fixed, references)
    after = {name: file_hash(path) for name, path in inputs.items()}
    prior_after = {str(p): file_hash(p) for p in prior_paths}
    if before != after or prior_hashes != prior_after:
        raise RuntimeError("Frozen input or previous result files changed during temporal evaluation")
    provenance.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
        sha256_after=after, prior_result_sha256_after=prior_after,
        fit_warnings=results[0][["model", "fit_warnings"]].to_dict(orient="records"))
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    return results

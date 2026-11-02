"""Fixed-model, fixed-threshold leave-one-development-cycle-out robustness."""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.data.load_nhanes import DEVELOPMENT_CYCLES, require_columns, validate_cohort
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_baseline_comparison import MODELS
from src.evaluation.nhanes_calibration_comparison import make_provisional_pipeline, validate_predictions
from src.evaluation.nhanes_preprocessing_comparison import DEVELOPMENT_PATH, OUTPUT_DIR, load_development_data
from src.evaluation.nhanes_threshold_analysis import threshold_metrics

THRESHOLD_PATH = OUTPUT_DIR / "threshold_candidate_points.csv"
FIXED_POINTS = ("default_0.50", "nearest_sensitivity_0.80", "nearest_sensitivity_0.90", "youden_j_maximum")
FREE_METRICS = ("roc_auc", "average_precision", "brier_score", "log_loss",
                "mean_predicted_probability", "predicted_probability_sd")
THRESHOLD_RATES = ("sensitivity", "specificity", "precision", "negative_predictive_value",
                   "balanced_accuracy", "accuracy", "f1", "predicted_positive_fraction",
                   "false_positive_rate", "false_negative_rate",
                   "sensitivity_target_deviation", "absolute_sensitivity_target_deviation")


def load_fixed_thresholds(path=THRESHOLD_PATH):
    """Load prior full-development decisions; no selection/recomputation allowed."""
    points = pd.read_csv(path, float_precision="round_trip")
    require_columns(points, ["model", "operating_point", "threshold", "target_sensitivity"], "fixed thresholds")
    if set(points.model) != set(MODELS):
        raise ValueError("Fixed threshold file must contain exactly the two specified models")
    selected = points[points.operating_point.isin(FIXED_POINTS)].copy()
    if selected.duplicated(["model", "operating_point"]).any() or len(selected) != 8:
        raise ValueError("Expected exactly one saved threshold per model and fixed operating point")
    for model in MODELS:
        group = selected[selected.model == model].set_index("operating_point")
        if set(group.index) != set(FIXED_POINTS):
            raise ValueError(f"Missing fixed operating points for {model}")
        if group.loc["default_0.50", "threshold"] != 0.5:
            raise ValueError("Saved default threshold must be exactly 0.5")
        for point, target in [("nearest_sensitivity_0.80", 0.8), ("nearest_sensitivity_0.90", 0.9)]:
            if group.loc[point, "target_sensitivity"] != target:
                raise ValueError(f"Unexpected saved sensitivity target for {point}")
    if not np.isfinite(selected.threshold).all() or not selected.threshold.between(0, 1).all():
        raise ValueError("Saved thresholds must be finite probabilities")
    return selected[["model", "operating_point", "threshold", "target_sensitivity"]].reset_index(drop=True)


def summarize_cycles(metrics, thresholds):
    """Equal weight per held-out cycle; sample SD (ddof=1), range not clinical flag."""
    rows = []
    for scope, frame, grouping, names in [
        ("threshold_free", metrics, ["model"], FREE_METRICS),
        ("fixed_threshold", thresholds, ["model", "operating_point"], THRESHOLD_RATES),
    ]:
        for keys, group in frame.groupby(grouping, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            identifiers = dict(zip(grouping, keys))
            for name in names:
                values = group[name]
                if name in ("sensitivity_target_deviation", "absolute_sensitivity_target_deviation") and values.isna().all():
                    continue  # No intended sensitivity target for default or Youden.
                rows.append(dict(scope=scope, **identifiers, metric=name, cycles_n=len(group),
                    valid_cycles_n=int(values.notna().sum()), mean=values.mean(skipna=False),
                    std=values.std(ddof=1, skipna=False), minimum=values.min(skipna=False),
                    maximum=values.max(skipna=False), range=values.max(skipna=False)-values.min(skipna=False)))
    return pd.DataFrame(rows)


def compare_cycles(df, fixed_thresholds, output_dir=OUTPUT_DIR):
    validate_cohort(df, "development")
    if set(df.nhanes_cycle) != set(DEVELOPMENT_CYCLES):
        raise ValueError("Leave-one-cycle-out requires all four development cycles exactly")
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError("Cannot save outputs into frozen processed data")
    metrics_rows, threshold_frames = [], []
    for cycle in DEVELOPMENT_CYCLES:
        train = df[df.nhanes_cycle != cycle]
        test = df[df.nhanes_cycle == cycle]
        if train.seqn.isin(test.seqn).any():
            raise ValueError("SEQN overlap between training and held-out cycle")
        if set(train.diabetes.unique()) != {0, 1} or set(test.diabetes.unique()) != {0, 1}:
            raise ValueError(f"Both outcome classes required in training and test for {cycle}")
        y = test.diabetes.astype(int)
        common = dict(held_out_cycle=cycle, train_n=len(train), test_n=len(test),
            positive_n=int(y.sum()), negative_n=int((y == 0).sum()), prevalence=float(y.mean()),
            training_cycles=json.dumps([c for c in DEVELOPMENT_CYCLES if c != cycle]))
        for model in MODELS:
            print(f"Held-out cycle {cycle}: {model}", flush=True)
            pipeline = make_provisional_pipeline(model)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(train[CORE_FEATURES], train.diabetes.astype(int))
                probability = pipeline.predict_proba(test[CORE_FEATURES])[:, 1]
            _, probability = validate_predictions(y, probability)
            for warning in caught:
                warnings.warn(f"{model}, cycle {cycle}: {warning.message}", warning.category, stacklevel=2)
            eps = np.finfo(np.float64).eps
            metrics_rows.append(dict(model=model, **common,
                roc_auc=roc_auc_score(y, probability), average_precision=average_precision_score(y, probability),
                brier_score=brier_score_loss(y, probability),
                log_loss=log_loss(y, np.clip(probability, eps, 1-eps), labels=[0, 1]),
                mean_predicted_probability=float(probability.mean()),
                predicted_probability_sd=float(probability.std(ddof=1)),
                fit_warnings=json.dumps([str(w.message) for w in caught])))
            fixed = fixed_thresholds[fixed_thresholds.model == model].reset_index(drop=True)
            scores = threshold_metrics(y, probability, fixed.threshold.to_numpy())
            scores["operating_point"] = fixed.operating_point
            scores["target_sensitivity"] = fixed.target_sensitivity
            scores["sensitivity_target_deviation"] = scores.sensitivity - scores.target_sensitivity
            scores["absolute_sensitivity_target_deviation"] = scores.sensitivity_target_deviation.abs()
            threshold_frames.append(scores.assign(model=model, **common))
    metrics = pd.DataFrame(metrics_rows)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    summary = summarize_cycles(metrics, thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("metrics", metrics), ("thresholds", thresholds), ("summary", summary)]:
        frame.to_csv(output_dir / f"cycle_robustness_{name}.csv", index=False)
    return metrics, thresholds, summary


def run_cycle_robustness():
    before = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    threshold_before = hashlib.sha256(THRESHOLD_PATH.read_bytes()).hexdigest()
    fixed = load_fixed_thresholds(THRESHOLD_PATH)
    results = compare_cycles(load_development_data(), fixed)
    after = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    threshold_after = hashlib.sha256(THRESHOLD_PATH.read_bytes()).hexdigest()
    if before != after or threshold_before != threshold_after:
        raise RuntimeError("Frozen development data or prior threshold results changed during robustness analysis")
    # Preserve Python float precision and encode absent targets as JSON null.
    saved_points = [{k: None if pd.isna(v) else v for k, v in record.items()}
                    for record in fixed.to_dict(orient="records")]
    provenance = dict(development_path=str(DEVELOPMENT_PATH), sha256_before=before, sha256_after=after,
        threshold_source=str(THRESHOLD_PATH), threshold_sha256_before=threshold_before,
        threshold_sha256_after=threshold_after, fixed_operating_points=saved_points,
        cycle_column="nhanes_cycle", held_out_cycles=list(DEVELOPMENT_CYCLES), random_seed=42,
        features=CORE_FEATURES, preprocessing="A_median", class_weight="none", calibration="none",
        no_tuning=True, no_threshold_selection=True, probability_sd_ddof=1, summary_sd_ddof=1,
        summary_weighting="equal weight per held-out cycle", prediction_rule="probability >= fixed threshold",
        bp_protocol={c: "oscillometric" if c == "2017_2020" else "legacy" for c in DEVELOPMENT_CYCLES},
        limitations="Previously selected development thresholds/hyperparameters include held-out-cycle information; conditional robustness, not fully independent validation",
        package_versions={n: version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost")},
        model_parameters={m: {k: "NaN" if isinstance(v, float) and np.isnan(v) else v
            for k, v in make_provisional_pipeline(m).named_steps["model"].get_params().items()} for m in MODELS},
        nonfinite_parameter_encoding="NaN model-parameter sentinel encoded as string NaN",
        fit_warnings=results[0][["model", "held_out_cycle", "fit_warnings"]].to_dict(orient="records"))
    (OUTPUT_DIR / "cycle_robustness_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    return results

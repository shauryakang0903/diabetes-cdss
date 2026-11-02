"""Descriptive development-OOF operating points, never clinical cutoff selection."""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.load_nhanes import validate_cohort
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_baseline_comparison import MODELS
from src.evaluation.nhanes_calibration_comparison import (
    make_provisional_pipeline, validate_predictions,
)
from src.evaluation.nhanes_preprocessing_comparison import (
    DEVELOPMENT_PATH, OUTPUT_DIR, load_development_data,
)

SEED = 42
GRID = np.arange(10, 991, dtype=float) / 1000  # 0.010 to 0.990 inclusive, step 0.001
POINTS = ("default_0.50", "nearest_sensitivity_0.80", "nearest_sensitivity_0.90",
          "youden_j_maximum", "nearest_sensitivity_specificity")


def safe_ratio(numerator, denominator):
    """Undefined ratios are missing, not misleading zeros (notably PPV/NPV)."""
    return np.divide(numerator, denominator,
                     out=np.full(np.shape(denominator), np.nan, dtype=float),
                     where=np.asarray(denominator) != 0)


def threshold_metrics(y, probability, thresholds):
    """Vectorized exact counts for prediction p >= threshold, including ties."""
    y, probability = validate_predictions(y, probability)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or not (
            (thresholds >= 0) & (thresholds <= 1)).all():
        raise ValueError("Thresholds must be nonempty, finite, one-dimensional, and in [0,1]")
    order = np.argsort(probability, kind="stable")
    prefix_positive = np.r_[0, np.cumsum(y[order], dtype=np.int64)]
    below = np.searchsorted(probability[order], thresholds, side="left")
    fn = prefix_positive[below]
    tn = below - fn
    tp = int(y.sum()) - fn
    fp = len(y) - below - tp
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    return pd.DataFrame(dict(
        threshold=thresholds, tp=tp, fp=fp, tn=tn, fn=fn,
        sensitivity=sensitivity, specificity=specificity,
        precision=safe_ratio(tp, tp + fp), negative_predictive_value=safe_ratio(tn, tn + fn),
        accuracy=(tp + tn) / len(y), f1=safe_ratio(2 * tp, 2 * tp + fp + fn),
        balanced_accuracy=(sensitivity + specificity) / 2,
        false_positive_rate=1 - specificity, false_negative_rate=1 - sensitivity,
        predicted_positive_fraction=(tp + fp) / len(y), youden_j=sensitivity + specificity - 1,
    ))


def candidate_operating_points(y, probability):
    """Scan attainable OOF score cutpoints, not merely the display grid.

    Primary selection: nearest absolute target error, maximum Youden J, or
    nearest sensitivity/specificity equality. Ties within 1e-12 favor higher
    specificity, then the larger threshold. The default remains exactly 0.50.
    Boundaries 0 and 1 are included, but out-of-range thresholds are never used.
    """
    all_points = threshold_metrics(y, probability, np.unique(np.r_[0.0, 0.5, 1.0, probability]))
    rows = []
    for name in POINTS:
        target = np.nan
        if name == "default_0.50":
            selected = all_points[all_points.threshold == 0.5]
            error = np.nan
        else:
            if name.startswith("nearest_sensitivity_0."):
                target = 0.8 if name.endswith("0.80") else 0.9
                errors = (all_points.sensitivity - target).abs()
            elif name == "youden_j_maximum":
                errors = all_points.youden_j.max() - all_points.youden_j
            else:
                errors = (all_points.sensitivity - all_points.specificity).abs()
            error = errors.min()
            selected = all_points[np.isclose(errors, error, rtol=0, atol=1e-12)]
        chosen = selected.sort_values(["specificity", "threshold"], ascending=False).iloc[0].to_dict()
        for count in ("tp", "fp", "tn", "fn"):
            chosen[count] = int(chosen[count])
        rows.append(dict(operating_point=name, target_sensitivity=target,
                         selection_error=error, **chosen))
    return pd.DataFrame(rows)


def generate_oof_predictions(df):
    validate_cohort(df, "development")
    if set(df.diabetes.unique()) != {0, 1} or df.diabetes.value_counts().min() < 5:
        raise ValueError("Five-fold OOF prediction requires at least five of each outcome")
    X, y = df[CORE_FEATURES], df.diabetes.astype(int)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    rows, diagnostics = [], []
    for model in MODELS:
        seen = np.zeros(len(df), dtype=int)
        for fold, (train, test) in enumerate(splits, 1):
            print(f"OOF fold {fold}/5: {model}", flush=True)
            pipeline = make_provisional_pipeline(model)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(X.iloc[train], y.iloc[train])
                probability = pipeline.predict_proba(X.iloc[test])[:, 1]
            validate_predictions(y.iloc[test], probability)
            for warning in caught:
                warnings.warn(f"{model} fold {fold}: {warning.message}", warning.category, stacklevel=2)
            seen[test] += 1
            frame = df.iloc[test][["seqn", "nhanes_cycle", "diabetes"]].copy()
            frame["model"], frame["outer_fold"], frame["probability"] = model, fold, probability
            rows.append(frame)
            diagnostics.append(dict(model=model, outer_fold=fold, train_n=len(train), test_n=len(test),
                train_positive_rate=float(y.iloc[train].mean()), test_positive_rate=float(y.iloc[test].mean()),
                warnings=[str(w.message) for w in caught]))
        if not np.all(seen == 1):
            raise RuntimeError(f"{model}: each participant must receive exactly one OOF prediction")
    oof = pd.concat(rows, ignore_index=True).sort_values(["model", "seqn"]).reset_index(drop=True)
    if oof.duplicated(["model", "seqn"]).any():
        raise RuntimeError("Duplicate participant OOF predictions within a model")
    return oof, diagnostics


def analyze_thresholds(df, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError("Cannot save outputs into frozen processed data")
    oof, diagnostics = generate_oof_predictions(df)
    curves, candidates = [], []
    for model in MODELS:
        group = oof[oof.model == model]
        curves.append(threshold_metrics(group.diabetes, group.probability, GRID).assign(model=model))
        candidates.append(candidate_operating_points(group.diabetes, group.probability).assign(model=model))
    curve, points = pd.concat(curves, ignore_index=True), pd.concat(candidates, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in [("threshold_curve", curve), ("threshold_candidate_points", points),
                            ("threshold_oof_predictions", oof)]:
        frame.to_csv(output_dir / f"{filename}.csv", index=False)
    return curve, points, oof, diagnostics


def run_threshold_analysis():
    before = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    results = analyze_thresholds(load_development_data())
    after = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Frozen development CSV changed during threshold analysis")
    provenance = dict(development_path=str(DEVELOPMENT_PATH), sha256_before=before, sha256_after=after,
        seed=SEED, outer_folds=5, shuffle=True, features=CORE_FEATURES,
        class_weight="none", calibration="none", preprocessing="A_median",
        prediction_rule="probability >= threshold", grid_start=0.01, grid_end=0.99, grid_step=0.001,
        candidate_thresholds="all unique OOF probabilities plus 0, 0.5, 1",
        candidate_tie_rule="within 1e-12 of minimum error, higher specificity then larger threshold",
        undefined_ppv_npv="NaN when denominator is zero",
        threshold_status="descriptive development candidates; no production cutoff selected",
        package_versions={n: version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost")},
        model_parameters={m: {k: "NaN" if isinstance(v, float) and np.isnan(v) else v
            for k, v in make_provisional_pipeline(m).named_steps["model"].get_params().items()} for m in MODELS},
        nonfinite_parameter_encoding="NaN model-parameter sentinel encoded as string NaN",
        fold_diagnostics=results[3])
    (OUTPUT_DIR / "threshold_analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    return results

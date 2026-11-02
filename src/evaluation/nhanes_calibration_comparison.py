"""Development-only, outer-held-out comparison of fixed models and calibration."""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from src.data.load_nhanes import validate_cohort
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_baseline_comparison import MODELS, make_baseline_pipeline
from src.evaluation.nhanes_preprocessing_comparison import (
    DEVELOPMENT_PATH, OUTPUT_DIR, load_development_data, score_predictions,
)

SEED = 42
METHODS = ("raw", "sigmoid", "isotonic")
LOGIT_EPSILON = 1e-6
BIN_EDGES = np.linspace(0.0, 1.0, 11)
METRICS = (
    "roc_auc", "average_precision", "brier_score", "log_loss",
    "calibration_intercept", "calibration_slope", "ece",
    "accuracy", "sensitivity", "specificity", "precision", "f1",
)


def make_provisional_pipeline(model):
    """Prespecified settings; reuse frozen baseline preprocessing without tuning."""
    pipeline = make_baseline_pipeline(model)
    if model == "LogisticRegression":
        pipeline.set_params(model__C=0.1)
    else:
        pipeline.set_params(model__n_estimators=200, model__learning_rate=0.03,
            model__max_depth=4, model__min_child_weight=10, model__subsample=0.8,
            model__colsample_bytree=0.8, model__reg_alpha=1.0, model__reg_lambda=10.0)
    return pipeline


def make_candidate(model, method, inner_splits):
    pipeline = make_provisional_pipeline(model)
    if method == "raw":
        return pipeline
    if method not in METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    # OOF scores train a single calibrator; a fresh base pipeline is then refit on
    # all outer-training rows. No prefit estimator or outer-test rows are supplied.
    return CalibratedClassifierCV(estimator=pipeline, method=method, cv=inner_splits,
                                  ensemble=False, n_jobs=1)


def validate_predictions(y, probability):
    y = np.asarray(y, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.ndim != 1 or y.shape != probability.shape or not len(y):
        raise ValueError("Expected nonempty matching one-dimensional targets/probabilities")
    if not np.isin(y, [0, 1]).all() or set(y) != {0, 1}:
        raise ValueError("Evaluation requires both binary outcome classes")
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return y, probability


def calibration_line(y, probability):
    """Joint unpenalized logistic diagnostic: logit P(Y=1)=a+b*logit(p).

    Ideal a=0, b=1. This joint intercept is NOT the offset-based calibration-in-
    the-large intercept. The regression is an evaluation diagnostic only; its
    coefficients are never used to alter evaluated predictions or choose a fit.
    """
    y, probability = validate_predictions(y, probability)
    clipped = np.clip(probability, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
    logits = np.log(clipped) - np.log1p(-clipped)
    if np.ptp(logits) < 1e-10:
        warnings.warn("Constant predictions: joint calibration line is not identifiable", UserWarning)
        return np.nan, np.nan
    design = np.column_stack([np.ones(len(y)), logits])

    def objective(beta):
        z = design @ beta
        loss = np.mean(np.logaddexp(0, z) - y * z)
        gradient = design.T @ (expit(z) - y) / len(y)
        return loss, gradient

    result = minimize(objective, np.array([0.0, 1.0]), jac=True, method="BFGS",
                      options={"gtol": 1e-8, "maxiter": 2000})
    fitted = expit(design @ result.x)
    hessian = design.T @ ((fitted * (1 - fitted))[:, None] * design) / len(y)
    if not result.success or not np.isfinite(result.x).all() or np.linalg.cond(hessian) > 1e12:
        warnings.warn(f"Unstable calibration-line diagnostic: {result.message}", UserWarning)
        return np.nan, np.nan
    return float(result.x[0]), float(result.x[1])


def reliability_bins(y, probability):
    """Ten fixed equal-width bins; [left,right), final bin includes p=1.

    Empty bins remain present (n=0, means missing). ECE=sum(n/N*|mean_p-rate|).
    Raw probabilities are used for bins, Brier, ECE, discrimination, thresholds.
    """
    y, probability = validate_predictions(y, probability)
    assignments = np.searchsorted(BIN_EDGES[1:-1], probability, side="right")
    rows = []
    for bin_id in range(10):
        mask = assignments == bin_id
        n = int(mask.sum())
        rows.append(dict(bin=bin_id, bin_left=BIN_EDGES[bin_id], bin_right=BIN_EDGES[bin_id + 1],
                         n=n, mean_probability=float(probability[mask].mean()) if n else np.nan,
                         observed_rate=float(y[mask].mean()) if n else np.nan))
    points = pd.DataFrame(rows)
    ece = ((points.n / len(y)) * (points.mean_probability - points.observed_rate).abs()).sum()
    return points, float(ece)


def evaluate_probabilities(y, probability):
    y, probability = validate_predictions(y, probability)
    metrics = score_predictions(y, probability)
    metrics["sensitivity"] = metrics.pop("recall")
    # Standard float64 machine-epsilon clipping for log loss, unlike the larger
    # 1e-6 clipping used ONLY to stabilize diagnostic logits.
    eps = np.finfo(np.float64).eps
    metrics["log_loss"] = log_loss(y, np.clip(probability, eps, 1 - eps), labels=[0, 1])
    metrics["calibration_intercept"], metrics["calibration_slope"] = calibration_line(y, probability)
    points, metrics["ece"] = reliability_bins(y, probability)
    return metrics, points


def compare_calibration(df, output_dir=OUTPUT_DIR):
    validate_cohort(df, "development")
    if set(df.diabetes.unique()) != {0, 1} or df.diabetes.value_counts().min() < 5:
        raise ValueError("Five-fold comparison requires at least five of each outcome")
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError("Cannot save results into frozen processed data")
    X, y = df[CORE_FEATURES], df.diabetes.astype(int)
    outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    metadata = dict(seed=SEED, inner_seed=SEED, outer_folds=5, inner_folds=3,
                    threshold=0.5, ensemble=False, feature_list=json.dumps(CORE_FEATURES),
                    logit_epsilon=LOGIT_EPSILON, ece_bins=10, class_weight="none")
    rows, curves, predictions = [], [], []
    for fold, (train, test) in enumerate(outer, 1):
        X_train, y_train = X.iloc[train], y.iloc[train]
        if y_train.value_counts().min() < 3:
            raise ValueError("Outer-training data cannot support three inner folds")
        inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train, y_train))
        for model in MODELS:
            for method in METHODS:
                candidate = f"{model}_{method}"
                print(f"Outer fold {fold}/5: {candidate}", flush=True)
                estimator = make_candidate(model, method, inner)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(X_train, y_train)
                    probability = estimator.predict_proba(X.iloc[test])[:, 1]
                    scores, points = evaluate_probabilities(y.iloc[test], probability)
                for warning in caught:
                    warnings.warn(f"{candidate}, fold {fold}: {warning.message}", warning.category, stacklevel=2)
                rows.append(dict(candidate=candidate, model=model, calibration=method, outer_fold=fold,
                    train_n=len(train), test_n=len(test), train_positive_rate=y_train.mean(),
                    test_positive_rate=y.iloc[test].mean(),
                    probability_zero_n=int(np.count_nonzero(probability == 0)),
                    probability_one_n=int(np.count_nonzero(probability == 1)),
                    warnings=json.dumps([str(w.message) for w in caught]), **metadata, **scores))
                curves.append(points.assign(candidate=candidate, model=model, calibration=method,
                                            scope="outer_fold", outer_fold=fold))
                predictions.append(pd.DataFrame(dict(candidate=candidate, model=model,
                    calibration=method, y=y.iloc[test].to_numpy(), probability=probability)))
    folds = pd.DataFrame(rows)
    summaries = []
    for (candidate, model, method), group in folds.groupby(["candidate", "model", "calibration"], sort=False):
        row = dict(candidate=candidate, model=model, calibration=method, **metadata)
        for metric in METRICS:
            # Never silently average fewer than five diagnostic values.
            row[metric + "_mean"] = group[metric].mean(skipna=False)
            row[metric + "_std"] = group[metric].std(ddof=1, skipna=False)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    for (candidate, model, method), group in pd.concat(predictions).groupby(
            ["candidate", "model", "calibration"], sort=False):
        points, _ = reliability_bins(group.y, group.probability)
        curves.append(points.assign(candidate=candidate, model=model, calibration=method,
                                    scope="pooled_outer_oof", outer_fold=0))
    curve_data = pd.concat(curves, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "calibration_comparison_folds.csv", index=False)
    summary.to_csv(output_dir / "calibration_comparison_summary.csv", index=False)
    curve_data.to_csv(output_dir / "calibration_curve_points.csv", index=False)
    return folds, summary, curve_data


def run_calibration_comparison():
    before = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    results = compare_calibration(load_development_data())
    after = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Frozen development CSV changed during calibration comparison")
    provenance = dict(development_path=str(DEVELOPMENT_PATH), sha256_before=before, sha256_after=after,
        seed=SEED, inner_seed=SEED, outer_folds=5, inner_folds=3, shuffle=True, n_jobs=1,
        ensemble=False, methods=METHODS, features=CORE_FEATURES, threshold=0.5,
        class_weight="none", preprocessing="A_median", logit_epsilon=LOGIT_EPSILON,
        log_loss_epsilon=float(np.finfo(np.float64).eps), bin_edges=BIN_EDGES.tolist(),
        calibration_line="Joint unpenalized logit P(Y=1)=intercept+slope*logit(clipped p)",
        package_versions={n: version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
        calibration_response={"LogisticRegression": "decision_function", "XGBoost": "predict_proba"},
        nonfinite_parameter_encoding="NaN sentinel encoded as the string NaN for strict JSON",
        model_parameters={m: {key: "NaN" if isinstance(value, float) and np.isnan(value) else value
            for key, value in make_provisional_pipeline(m).named_steps["model"].get_params().items()}
            for m in MODELS})
    (OUTPUT_DIR / "calibration_comparison_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    return results

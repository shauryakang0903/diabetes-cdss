"""Untuned, development-only LR/XGBoost comparison."""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_nhanes import validate_cohort
from src.data.nhanes_model_preprocessing import make_preprocessor
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_preprocessing_comparison import (
    DEVELOPMENT_PATH, OUTPUT_DIR, METRICS, load_development_data, score_predictions,
)

MODELS = ("LogisticRegression", "XGBoost")
BASELINE_METRICS = tuple("sensitivity" if m == "recall" else m for m in METRICS)


def make_baseline_pipeline(model_name):
    if model_name == "LogisticRegression":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                   class_weight=None, random_state=42)
        branch = "logistic"
    elif model_name == "XGBoost":
        # Fixed conventional baseline, no search, weights, eval_set, or early stopping.
        model = XGBClassifier(
            objective="binary:logistic", n_estimators=100, learning_rate=0.3,
            max_depth=6, min_child_weight=1, gamma=0, subsample=1.0,
            colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0,
            scale_pos_weight=1.0, tree_method="hist", device="cpu",
            eval_metric="logloss", random_state=42, n_jobs=1,
        )
        branch = "xgboost"
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    return Pipeline([
        ("preprocessing", make_preprocessor("A_median", branch=branch)),
        ("model", model),
    ])


def compare_baselines(df, output_dir=OUTPUT_DIR):
    """Fit fresh complete pipelines on identical five development-only folds."""
    validate_cohort(df, "development")
    if set(df.diabetes.unique()) != {0, 1} or df.diabetes.value_counts().min() < 5:
        raise ValueError("Five-fold comparison requires at least five of each outcome")
    X, y = df[CORE_FEATURES], df.diabetes.astype(int)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
    metadata = {
        "random_seed": 42, "fold_count": 5, "threshold": 0.5,
        "preprocessing_strategy": "A_median", "feature_list": json.dumps(CORE_FEATURES),
        "class_weight": "none",
        "package_versions": json.dumps({n: version(n) for n in
                                      ("numpy", "pandas", "scikit-learn", "scipy", "xgboost")}),
    }
    rows = []
    for name in MODELS:
        for fold, (train, held_out) in enumerate(splits, 1):
            pipeline = make_baseline_pipeline(name)
            # Includes estimator defaults; resolved booster config is saved after fit.
            params = json.dumps(pipeline.named_steps["model"].get_params(),
                                sort_keys=True, allow_nan=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(X.iloc[train], y.iloc[train])
            for warning in caught:
                warnings.warn(f"{name}, fold {fold}: {warning.message}",
                              warning.category, stacklevel=2)
            probabilities = pipeline.predict_proba(X.iloc[held_out])[:, 1]
            scores = score_predictions(y.iloc[held_out], probabilities)
            scores["sensitivity"] = scores.pop("recall")
            rows.append({
                "model": name, "fold": fold, **metadata, "model_parameters": params,
                "resolved_booster_config": (
                    pipeline.named_steps["model"].get_booster().save_config()
                    if name == "XGBoost" else ""),
                "train_n": len(train), "validation_n": len(held_out),
                "train_positive_rate": y.iloc[train].mean(),
                "validation_positive_rate": y.iloc[held_out].mean(),
                "fit_warnings": json.dumps([str(w.message) for w in caught]),
                **scores,
            })
    folds = pd.DataFrame(rows)
    summaries = []
    for name, group in folds.groupby("model", sort=False):
        row = {"model": name, **metadata,
               "model_parameters": group.model_parameters.iloc[0]}
        for metric in BASELINE_METRICS:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    output_dir = Path(output_dir)
    if output_dir.resolve() == DEVELOPMENT_PATH.parent.resolve():
        raise ValueError("Results must not be written into frozen processed data")
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "baseline_model_comparison_folds.csv", index=False)
    summary.to_csv(output_dir / "baseline_model_comparison_summary.csv", index=False)
    return folds, summary


def run_baseline_comparison():
    """Read only the fixed development file; verify its bytes remain unchanged."""
    before = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    result = compare_baselines(load_development_data())
    after = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Frozen development CSV changed during the experiment")
    (OUTPUT_DIR / "baseline_model_comparison_provenance.json").write_text(
        json.dumps({"development_path": str(DEVELOPMENT_PATH),
                    "sha256_before": before, "sha256_after": after}, indent=2),
        encoding="utf-8",
    )
    return result

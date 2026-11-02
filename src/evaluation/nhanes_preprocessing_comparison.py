"""Development-only probe experiment, not final model selection."""

import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.load_nhanes import validate_cohort
from src.data.nhanes_model_preprocessing import STRATEGIES, make_preprocessor
from src.data.preprocessing import CORE_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_PATH = PROJECT_ROOT / "data/processed/nhanes/nhanes_development_2011_2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/nhanes"
METRICS = ("roc_auc", "average_precision", "accuracy", "recall", "specificity",
           "precision", "f1", "brier_score")


def load_development_data():
    """Fixed development input: deliberately offers no alternate cohort path."""
    df = pd.read_csv(DEVELOPMENT_PATH)
    validate_cohort(df, "development")
    if len(df) != 25580 or int(df.diabetes.sum()) != 3690:
        raise ValueError("Frozen development cohort counts do not match specification")
    return df


def make_probe_pipeline(strategy, random_seed=42):
    return Pipeline([
        ("preprocessing", make_preprocessor(strategy, random_seed=random_seed)),
        # Primary probe uses observed prevalence without class weighting. Balanced
        # weights change the fitted probability objective and confound this comparison.
        ("model", LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000,
                                     class_weight=None, random_state=random_seed)),
    ])


def score_predictions(y_true, probabilities):
    """Score only held-out predictions; use a fixed, untuned threshold of 0.5."""
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    if tn + fp == 0:
        raise ValueError("Specificity requires negative outcomes in the held-out fold")
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        # Average precision is the requested PR summary, not trapezoidal PR area.
        "average_precision": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def compare_preprocessing(df, output_dir=OUTPUT_DIR, random_seed=42, fold_count=5):
    """Use identical stratified folds for both strategies; fit fresh full pipelines.

    No full-cohort imputation, encoding, scaling, threshold tuning, or refitting.
    Fold SD is descriptive variability, not an independent-sample standard error.
    """
    validate_cohort(df, "development")
    if set(df.diabetes.unique()) != {0, 1}:
        raise ValueError("Cross-validation requires both binary outcome classes")
    if fold_count < 2 or df.diabetes.value_counts().min() < fold_count:
        raise ValueError("Each outcome class must support the requested stratified folds")
    X, y = df[CORE_FEATURES], df.diabetes.astype(int)
    splits = list(StratifiedKFold(n_splits=fold_count, shuffle=True,
                                random_state=random_seed).split(X, y))
    metadata = {"random_seed": random_seed, "fold_count": fold_count,
                "feature_list": json.dumps(CORE_FEATURES), "threshold": 0.5,
                "class_weight": "none",
                "package_versions": json.dumps({name: version(name) for name in
                                     ("numpy", "pandas", "scikit-learn", "scipy")})}
    rows = []
    for strategy in STRATEGIES:
        for fold, (train, test) in enumerate(splits, 1):
            pipeline = make_probe_pipeline(strategy, random_seed)
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always")
                pipeline.fit(X.iloc[train], y.iloc[train])
            # Retain diagnostics in the artifact and surface them to the runner.
            for warning in fit_warnings:
                warnings.warn(f"{strategy}, fold {fold}: {warning.message}",
                              warning.category, stacklevel=2)
            probabilities = pipeline.predict_proba(X.iloc[test])[:, 1]
            rows.append({"strategy": strategy, "fold": fold, **metadata,
                         "train_n": len(train), "test_n": len(test),
                         "test_positive_n": int(y.iloc[test].sum()),
                         "fit_warnings": json.dumps([str(w.message) for w in fit_warnings]),
                         **score_predictions(y.iloc[test], probabilities)})
    folds = pd.DataFrame(rows)
    summaries = []
    for strategy, group in folds.groupby("strategy", sort=False):
        row = {"strategy": strategy, **metadata}
        for metric in METRICS:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "preprocessing_comparison_folds.csv", index=False)
    summary.to_csv(output_dir / "preprocessing_comparison_summary.csv", index=False)
    return folds, summary

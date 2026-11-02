"""Nested ROC-AUC tuning restricted to development data; no final model refit."""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.data.load_nhanes import validate_cohort
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_baseline_comparison import (
    BASELINE_METRICS, MODELS, make_baseline_pipeline,
)
from src.evaluation.nhanes_preprocessing_comparison import (
    DEVELOPMENT_PATH, OUTPUT_DIR, load_development_data, score_predictions,
)

SEED = 42
INNER_SEED = 42
XGB_ITERATIONS = 20
LR_SPACE = {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
XGB_SPACE = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.1, 0.3],
    "model__max_depth": [2, 3, 4, 6],
    "model__min_child_weight": [1, 5, 10],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__reg_alpha": [0.0, 0.1, 1.0],
    "model__reg_lambda": [1.0, 5.0, 10.0],
}


def make_search(model, inner_splits):
    """Fresh pipeline cloned independently for each candidate and inner split.

    Fixed lbfgs with L2 (l1_ratio=0 in sklearn 1.9) avoids invalid solver/penalty
    combinations and deprecated explicit penalty arguments. Only C is searched.
    The same 20 randomized candidates are used in each outer fold (seed 42).
    """
    pipeline = make_baseline_pipeline(model)
    common = dict(estimator=pipeline, cv=inner_splits, scoring="roc_auc",
                  refit=True, n_jobs=1, error_score="raise", return_train_score=False)
    if model == "LogisticRegression":
        return GridSearchCV(param_grid=LR_SPACE, **common)
    if model == "XGBoost":
        return RandomizedSearchCV(param_distributions=XGB_SPACE, n_iter=XGB_ITERATIONS,
                                  random_state=SEED, **common)
    raise ValueError(f"Unknown model: {model}")


def parameter_stability(folds):
    """Long-format selections, LR frequencies, and descriptive XGB variability.

    Predeclared substantial-variation flag: at least four of eight XGB tuned
    parameters take two or more selected values over the five outer folds.
    This is a descriptive heuristic, not a significance test.
    """
    rows = []
    xgb = folds[folds.model == "XGBoost"]
    selections = [json.loads(p) for p in xgb.selected_hyperparameters]
    varying = sum(len({s[key] for s in selections}) > 1 for key in XGB_SPACE)
    for _, row in folds.iterrows():
        for param, value in json.loads(row.selected_hyperparameters).items():
            rows.append(dict(record_type="selection", model=row.model,
                             outer_fold=row.outer_fold, parameter=param, value=value,
                             selection_count=1, distinct_values=None,
                             substantial_variation=(varying >= 4 if row.model == "XGBoost" else False)))
    lr_counts = folds.loc[folds.model == "LogisticRegression", "selected_hyperparameters"].value_counts()
    # Include zero counts for evaluated LR settings, not just winners.
    for c in LR_SPACE["model__C"]:
        setting = json.dumps({"model__C": c}, sort_keys=True)
        rows.append(dict(record_type="frequency", model="LogisticRegression",
                         outer_fold=None, parameter="model__C", value=c,
                         selection_count=int(lr_counts.get(setting, 0)),
                         distinct_values=None, substantial_variation=False))
    for key in XGB_SPACE:
        rows.append(dict(record_type="variability", model="XGBoost", outer_fold=None,
                         parameter=key, value=None, selection_count=None,
                         distinct_values=len({s[key] for s in selections}),
                         substantial_variation=varying >= 4))
    return pd.DataFrame(rows)


def nested_tuning(df, output_dir=OUTPUT_DIR):
    validate_cohort(df, "development")
    if set(df.diabetes.unique()) != {0, 1} or df.diabetes.value_counts().min() < 5:
        raise ValueError("Nested CV requires at least five of each binary outcome")
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError("Cannot write experiment outputs into frozen processed data")
    X, y = df[CORE_FEATURES], df.diabetes.astype(int)
    outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    metadata = dict(random_seed=SEED, inner_seed=INNER_SEED, outer_folds=5, inner_folds=3,
                    threshold=0.5, feature_list=json.dumps(CORE_FEATURES),
                    class_weight="none", preprocessing="A_median",
                    package_versions=json.dumps({n: version(n) for n in
                        ("numpy", "pandas", "scikit-learn", "scipy", "xgboost")}))
    rows, audits = [], []
    for fold, (train, test) in enumerate(outer, 1):
        X_train, y_train = X.iloc[train], y.iloc[train]
        if y_train.value_counts().min() < 3:
            raise ValueError("Outer training cohort cannot support three stratified inner folds")
        inner = list(StratifiedKFold(3, shuffle=True, random_state=INNER_SEED).split(X_train, y_train))
        for name in MODELS:
            print(f"Outer fold {fold}/5: tuning {name}", flush=True)
            search = make_search(name, inner)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                search.fit(X_train, y_train)
            for warning in caught:
                warnings.warn(f"{name} fold {fold}: {warning.message}", warning.category, stacklevel=2)
            probability = search.best_estimator_.predict_proba(X.iloc[test])[:, 1]
            scores = score_predictions(y.iloc[test], probability)
            scores["sensitivity"] = scores.pop("recall")
            rows.append(dict(model=name, outer_fold=fold, train_n=len(train), test_n=len(test),
                train_positive_rate=y_train.mean(), test_positive_rate=y.iloc[test].mean(),
                best_inner_roc_auc=search.best_score_,
                selected_hyperparameters=json.dumps(search.best_params_, sort_keys=True),
                model_parameters=json.dumps(search.best_estimator_.named_steps["model"].get_params(), sort_keys=True),
                fit_warnings=json.dumps([str(w.message) for w in caught]),
                **metadata, **scores))
            results = search.cv_results_
            for i, params in enumerate(results["params"]):
                audits.append(dict(model=name, outer_fold=fold,
                    parameters=json.dumps(params, sort_keys=True),
                    mean_inner_roc_auc=results["mean_test_score"][i],
                    std_inner_roc_auc=results["std_test_score"][i],
                    rank=int(results["rank_test_score"][i])))
    folds = pd.DataFrame(rows)
    summary_rows = []
    for name, group in folds.groupby("model", sort=False):
        row = dict(model=name, **metadata)
        for metric in BASELINE_METRICS:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    stability = parameter_stability(folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("outer_folds", folds), ("summary", summary),
                        ("hyperparameters", stability), ("inner_search", pd.DataFrame(audits))]:
        frame.to_csv(output_dir / f"nested_tuning_{name}.csv", index=False)
    return folds, summary, stability


def run_nested_tuning():
    before = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    results = nested_tuning(load_development_data())
    after = hashlib.sha256(DEVELOPMENT_PATH.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Frozen development CSV changed during nested tuning")
    (OUTPUT_DIR / "nested_tuning_provenance.json").write_text(json.dumps(dict(
        development_path=str(DEVELOPMENT_PATH), sha256_before=before, sha256_after=after,
        lr_search=LR_SPACE, xgb_search=XGB_SPACE, xgb_n_iter=XGB_ITERATIONS,
        inner_seed=INNER_SEED, outer_seed=SEED, search_seed=SEED), indent=2), encoding="utf-8")
    return results

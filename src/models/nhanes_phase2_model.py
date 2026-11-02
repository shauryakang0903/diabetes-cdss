"""Build/load the frozen Phase-2 artifact; inference helpers never fit anything.

Load joblib artifacts only from trusted sources: a checksum verifies integrity,
not the trustworthiness of pickle content. Use the recorded training environment.
"""

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.data.nhanes_model_preprocessing import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_preprocessor,
)
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_cycle_robustness import load_fixed_thresholds
from src.evaluation.nhanes_preprocessing_comparison import (
    DEVELOPMENT_PATH, OUTPUT_DIR as RESULTS_DIR, load_development_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models/nhanes"
ARTIFACT_NAME = "phase2_xgboost_pipeline.joblib"
METADATA_NAME = "phase2_xgboost_metadata.json"
DEFAULT_ARTIFACT_PATH = MODEL_DIR / ARTIFACT_NAME
DEFAULT_METADATA_PATH = MODEL_DIR / METADATA_NAME
MODEL_VERSION = "nhanes_phase2_xgb_v1"
THRESHOLD_PATH = RESULTS_DIR / "threshold_candidate_points.csv"
TEMPORAL_METRICS_PATH = RESULTS_DIR / "temporal_validation_metrics.csv"
TEMPORAL_COMPLETENESS_PATH = RESULTS_DIR / "temporal_validation_missingness_robustness.csv"
PACKAGES = ("scikit-learn", "xgboost", "pandas", "numpy", "joblib", "scipy")
REFERENCE_NAMES = {
    "default": "default_0.50",
    "sensitivity_80_reference": "nearest_sensitivity_0.80",
    "sensitivity_90_reference": "nearest_sensitivity_0.90",
    "youden_reference": "youden_j_maximum",
}
FROZEN_XGB_PARAMETERS = dict(
    n_estimators=200, learning_rate=0.03, max_depth=4, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=10.0,
    objective="binary:logistic", scale_pos_weight=1.0, gamma=0,
    tree_method="hist", device="cpu", eval_metric="logloss", n_jobs=1,
    random_state=42,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_package_version(name):
    """Official Linux CPU wheels use a different distribution name, same API/version."""
    try:
        return version(name)
    except PackageNotFoundError:
        if name == "xgboost":
            return version("xgboost-cpu")
        raise


def make_frozen_pipeline():
    """Stable artifact specification, independent of provisional model factories."""
    return Pipeline([
        ("preprocessing", make_preprocessor("A_median", branch="xgboost", random_seed=42)),
        ("model", XGBClassifier(**FROZEN_XGB_PARAMETERS)),
    ])


def validate_input_columns(data):
    """Require harmonized Core-10 names/units and numeric NHANES category codes.

    Extras (including IDs/target) are ignored. Original feature names/order are
    preserved in the returned frame. Missing values remain missing: no learned
    preprocessing or imputation takes place here. Numeric-string coercion is
    input type validation only; free-text category labels are not recoded.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame with named Core-10 columns")
    if data.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names")
    missing = [c for c in CORE_FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required Core-10 columns: {missing}")
    if data.empty:
        raise ValueError("Input contains no participants")
    try:
        result = data.loc[:, CORE_FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError("Core-10 values must be numeric, using harmonized NHANES category codes") from error
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite predictors are unsupported; use missing values for unknown measurements")
    return result


def load_metadata(metadata_path=DEFAULT_METADATA_PATH):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Unsupported frozen model version: {metadata.get('model_version')}")
    if metadata.get("core_features") != CORE_FEATURES:
        raise ValueError("Artifact metadata Core-10 schema does not match the expected feature names")
    thresholds = metadata.get("reference_thresholds", {})
    if set(thresholds) != set(REFERENCE_NAMES) or not all(
            isinstance(t, (int, float)) and np.isfinite(t) and 0 <= t <= 1 for t in thresholds.values()):
        raise ValueError("Metadata must contain four valid frozen reference operating points")
    return metadata


def load_frozen_model(artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=None):
    """Load a trusted fitted pipeline after SHA/schema checks; warn on version drift."""
    artifact_path = Path(artifact_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else artifact_path.with_name(METADATA_NAME)
    metadata = load_metadata(metadata_path)
    if sha256(artifact_path) != metadata.get("artifact_sha256"):
        raise ValueError("Frozen artifact SHA-256 does not match metadata")
    differences = [f"{name}: saved {metadata['package_versions'][name]}, installed {runtime_package_version(name)}"
                   for name in PACKAGES if metadata["package_versions"][name] != runtime_package_version(name)]
    if platform.python_version().split(".")[:2] != metadata["python_version"].split(".")[:2]:
        differences.append(f"Python: saved {metadata['python_version']}, installed {platform.python_version()}")
    if differences:
        warnings.warn("Frozen artifact environment differs; cross-version pickle compatibility is not guaranteed: "
                      + "; ".join(differences), UserWarning, stacklevel=2)
    pipeline = joblib.load(artifact_path)
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["preprocessing", "model"]:
        raise ValueError("Expected a complete frozen preprocessing + classifier pipeline")
    if not isinstance(pipeline.named_steps["model"], XGBClassifier):
        raise ValueError("Expected the frozen XGBoost classifier")
    check_is_fitted(pipeline)
    if list(pipeline.feature_names_in_) != CORE_FEATURES or list(pipeline.classes_) != [0, 1]:
        raise ValueError("Artifact feature schema or outcome class order is invalid")
    if list(pipeline.named_steps["preprocessing"].get_feature_names_out()) != metadata["transformed_feature_names"]:
        raise ValueError("Transformed artifact feature names do not match metadata")
    for name, expected in FROZEN_XGB_PARAMETERS.items():
        if pipeline.named_steps["model"].get_params()[name] != expected:
            raise ValueError(f"Frozen hyperparameter mismatch: {name}")
    return pipeline


def predict_probabilities(data, pipeline=None, *, artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=None):
    """Positive-class diagnosed-diabetes probabilities, never a clinical diagnosis."""
    if pipeline is None:
        pipeline = load_frozen_model(artifact_path, metadata_path)
    X = validate_input_columns(data)
    check_is_fitted(pipeline)
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError("Expected binary outcome classes in order [0, 1]")
    probability = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise RuntimeError("Frozen artifact produced invalid probabilities")
    return probability


def predict_at_reference(data, reference_name="default", *, artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=None):
    """Binary decisions at a named development reference, not a clinical cutoff."""
    metadata_path = Path(metadata_path) if metadata_path is not None else Path(artifact_path).with_name(METADATA_NAME)
    metadata = load_metadata(metadata_path)
    if reference_name not in REFERENCE_NAMES:
        raise ValueError(f"Unknown operating reference '{reference_name}'. Choose from {list(REFERENCE_NAMES)}")
    probability = predict_probabilities(data, artifact_path=artifact_path, metadata_path=metadata_path)
    return (probability >= metadata["reference_thresholds"][reference_name]).astype(int)


def build_frozen_artifact(output_dir=MODEL_DIR):
    """Fit exactly once from development; temporal result tables are documentation only.

    Existing v1 artifacts are never overwritten. Use a separately authorized
    versioned location for a later model. No temporal cohort path is accepted.
    """
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError("Cannot save model artifacts into frozen processed data")
    artifact_path, metadata_path = output_dir / ARTIFACT_NAME, output_dir / METADATA_NAME
    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError("Frozen artifact/metadata already exists; v1 will not be overwritten")
    inputs = dict(development=DEVELOPMENT_PATH, thresholds=THRESHOLD_PATH,
                  temporal_metrics_documentation=TEMPORAL_METRICS_PATH,
                  temporal_completeness_documentation=TEMPORAL_COMPLETENESS_PATH)
    hashes_before = {name: sha256(path) for name, path in inputs.items()}
    prior = {str(path): sha256(path) for path in RESULTS_DIR.iterdir() if path.is_file()}
    development = load_development_data()
    X = validate_input_columns(development)
    y = development.diabetes.astype(int)
    fixed = load_fixed_thresholds(THRESHOLD_PATH)
    fixed = fixed[fixed.model == "XGBoost"].set_index("operating_point")
    thresholds = {name: float(fixed.loc[point, "threshold"]) for name, point in REFERENCE_NAMES.items()}
    temporal = pd.read_csv(TEMPORAL_METRICS_PATH, float_precision="round_trip")
    temporal = temporal[temporal.model == "XGBoost"]
    if len(temporal) != 1:
        raise ValueError("Expected one previously saved XGBoost temporal metric row")
    temporal = temporal.iloc[0]
    documented_metrics = {name: float(temporal[name]) for name in
        ("roc_auc", "average_precision", "brier_score", "log_loss", "calibration_slope", "calibration_intercept", "ece")}
    if not np.isfinite(list(documented_metrics.values())).all():
        raise ValueError("Saved temporal documentation metrics must be finite")
    completeness = pd.read_csv(TEMPORAL_COMPLETENESS_PATH, float_precision="round_trip")
    completeness = completeness[completeness.model == "XGBoost"]
    started = datetime.now(timezone.utc).isoformat()
    pipeline = make_frozen_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X, y)  # The sole training call: development rows only.
    finished = datetime.now(timezone.utc).isoformat()
    for warning in caught:
        warnings.warn(f"Frozen artifact training: {warning.message}", warning.category, stacklevel=2)
    # Development-only serialization probe, plus an artificial all-missing row.
    # No predictive performance is evaluated and no temporal participants are read.
    probe = X.iloc[np.linspace(0, len(X)-1, min(128, len(X)), dtype=int)].copy()
    probe = pd.concat([probe, pd.DataFrame([{c: np.nan for c in CORE_FEATURES}])], ignore_index=True)
    before_save = predict_probabilities(probe, pipeline)
    transformed = pipeline.named_steps["preprocessing"].get_feature_names_out().tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path, compress=3)
    reloaded = joblib.load(artifact_path)  # Trusted artifact just created here.
    after_reload = predict_probabilities(probe, reloaded)
    np.testing.assert_allclose(before_save, after_reload, rtol=1e-7, atol=1e-8)
    hashes_after = {name: sha256(path) for name, path in inputs.items()}
    prior_after = {path: sha256(path) for path in prior}
    if hashes_before != hashes_after or prior != prior_after:
        raise RuntimeError("A frozen input or prior result changed during artifact creation")
    prep = pipeline.named_steps["preprocessing"]
    metadata = dict(
        metadata_schema_version=1, model_name="XGBoost", model_version=MODEL_VERSION,
        project_purpose="Explainable diabetes classification/risk-support research CDSS; probabilities are not clinical diagnoses",
        target_definition="diabetes: 1=physician-diagnosed diabetes from DIQ010; 0=no physician-diagnosed diabetes",
        development_cohort="NHANES 2011-2012, 2013-2014, 2015-2016, 2017-March 2020; adults age >=20 with valid DIQ010 binary outcome",
        development_n=len(development), positive_n=int(y.sum()), negative_n=int(y.eq(0).sum()),
        core_features=CORE_FEATURES, input_feature_count=len(CORE_FEATURES),
        transformed_feature_count=len(transformed), transformed_feature_names=transformed,
        preprocessing_specification=dict(numeric_features=NUMERIC_FEATURES,
            numeric_imputation="median", numeric_scaling="none", keep_empty_features=True,
            categorical_features=CATEGORICAL_FEATURES, categorical_imputation="most_frequent",
            encoding="OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            training_scope="development only", contained_in_saved_pipeline=True),
        learned_numeric_medians=prep.named_transformers_["numeric"].named_steps["imputer"].statistics_.tolist(),
        learned_categorical_modes=prep.named_transformers_["categorical"].named_steps["imputer"].statistics_.tolist(),
        learned_categories=[values.tolist() for values in prep.named_transformers_["categorical"].named_steps["encoder"].categories_],
        xgboost_hyperparameters={k: "NaN" if isinstance(v, float) and np.isnan(v) else v
                                for k, v in pipeline.named_steps["model"].get_params().items()},
        resolved_booster_configuration=json.loads(pipeline.named_steps["model"].get_booster().save_config()),
        class_weight="none", post_hoc_calibration="none", random_seed=42, fit_count=1,
        reference_thresholds=thresholds, threshold_source=str(THRESHOLD_PATH),
        threshold_statement="Development-derived operating references, not clinically validated cutoffs; no production clinical threshold selected",
        development_csv_path=str(DEVELOPMENT_PATH), development_csv_sha256=hashes_after["development"],
        threshold_results_sha256=hashes_after["thresholds"], artifact_path=str(artifact_path.resolve()),
        artifact_sha256=sha256(artifact_path), python_version=platform.python_version(),
        package_versions={name: version(name) for name in PACKAGES},
        training_started_utc=started, training_completed_utc=finished,
        fit_warnings=[str(w.message) for w in caught],
        temporal_validation_status="Primary frozen temporal evaluation completed previously; documentation only; no temporal cohort used or opened during artifact creation",
        temporal_validation_summary=dict(metrics=documented_metrics, source=str(TEMPORAL_METRICS_PATH),
            source_sha256=hashes_after["temporal_metrics_documentation"],
            temporal_n=int(temporal["temporal_n"]), temporal_positive_n=int(temporal["temporal_positive_n"]),
            temporal_negative_n=int(temporal["temporal_negative_n"]), role="documentation only, never used to fit or construct the predictive pipeline"),
        known_temporal_missingness_limitation=dict(
            description="Higher temporal predictor missingness and lower discrimination in participants with missing Core-10 fields; no adaptation performed",
            saved_result_source=str(TEMPORAL_COMPLETENESS_PATH),
            saved_result_sha256=hashes_after["temporal_completeness_documentation"],
            xgboost_completeness_results=completeness.to_dict(orient="records")),
        serialization=dict(format="joblib compressed pickle, compression=3", probe_n=len(probe),
            probe_scope="development-derived rows plus one artificial all-missing row", reload_equivalent=True,
            maximum_absolute_prediction_difference=float(np.max(np.abs(before_save-after_reload))), rtol=1e-7, atol=1e-8,
            trust_requirement="Load only trusted pickle/joblib files; a checksum is not authentication",
            compatibility="Use the recorded Python/library environment; cross-version loading is not guaranteed"),
        input_sha256_before=hashes_before, input_sha256_after=hashes_after,
        prior_result_sha256_before=prior, prior_result_sha256_after=prior_after,
        source_sha256={str(Path(__file__).resolve()): sha256(__file__),
            str(PROJECT_ROOT / "src/data/nhanes_model_preprocessing.py"): sha256(PROJECT_ROOT / "src/data/nhanes_model_preprocessing.py")},
        nonfinite_parameter_encoding="NaN XGBoost missing-value sentinel encoded as string NaN",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return artifact_path, metadata_path, metadata

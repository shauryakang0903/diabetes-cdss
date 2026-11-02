import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.preprocessing import CORE_FEATURES
from src.models import nhanes_phase2_model as frozen


@pytest.fixture
def built(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    n = 120
    dev = pd.DataFrame({c: rng.normal(50, 5, n) for c in CORE_FEATURES})
    dev["age"] = rng.integers(20, 80, n)
    for col, values in [("sex", [1., 2.]), ("race_ethnicity", [1., 3., 6.]), ("smoking_status", [0., 1., 2.])]:
        dev[col] = rng.choice(values, n)
    dev.loc[::4, ["bmi", "sex"]] = np.nan
    dev["seqn"] = np.arange(n)
    dev["nhanes_cycle"] = "2011_2012"
    dev["diabetes"] = np.tile([0, 0, 1], 40)
    dev_path = tmp_path / "data/processed/nhanes/development.csv"
    dev_path.parent.mkdir(parents=True)
    dev.to_csv(dev_path, index=False)
    refs = tmp_path / "results"
    refs.mkdir()
    threshold_path = refs / "threshold_candidate_points.csv"
    rows = []
    exact_values = {"default": 0.5, "sensitivity_80_reference": 0.14026810228824615,
                    "sensitivity_90_reference": 0.08646801114082336, "youden_reference": 0.12872706353664398}
    for model in ["LogisticRegression", "XGBoost"]:
        for name, point in frozen.REFERENCE_NAMES.items():
            target = 0.8 if name == "sensitivity_80_reference" else (0.9 if name == "sensitivity_90_reference" else np.nan)
            rows.append(dict(model=model, operating_point=point, target_sensitivity=target, threshold=exact_values[name]))
    pd.DataFrame(rows).to_csv(threshold_path, index=False)
    temporal_path = refs / "temporal_validation_metrics.csv"
    documented = dict(model="XGBoost", temporal_n=7536, temporal_positive_n=1070, temporal_negative_n=6466,
        roc_auc=0.812354197802452, average_precision=0.4247273958297281, brier_score=0.10001399529978304,
        log_loss=0.3252287354709952, calibration_slope=1.0918583561570374,
        calibration_intercept=0.13513768635391243, ece=0.006950486976801138)
    pd.DataFrame([documented]).to_csv(temporal_path, index=False)
    completeness_path = refs / "temporal_validation_missingness_robustness.csv"
    pd.DataFrame([dict(model="XGBoost", completeness_group="all_core10_observed", n=4969,
                      roc_auc=0.831413, average_precision=0.45184, brier_score=0.095843)]).to_csv(completeness_path, index=False)
    sentinel = tmp_path / "nhanes_validation_2021_2023.csv"
    sentinel.write_text("Do not read: temporal cohort sentinel", encoding="utf-8")
    initial_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (dev_path, threshold_path, temporal_path, completeness_path, sentinel)}
    monkeypatch.setattr(frozen, "DEVELOPMENT_PATH", dev_path)
    monkeypatch.setattr(frozen, "THRESHOLD_PATH", threshold_path)
    monkeypatch.setattr(frozen, "TEMPORAL_METRICS_PATH", temporal_path)
    monkeypatch.setattr(frozen, "TEMPORAL_COMPLETENESS_PATH", completeness_path)
    monkeypatch.setattr(frozen, "RESULTS_DIR", refs)
    calls, fitted_pipelines = [], []
    read = pd.read_csv
    def tracked_read(path, **kwargs):
        assert str(path) in {str(p) for p in (dev_path, threshold_path, temporal_path, completeness_path)}
        calls.append(str(path))
        return read(path, **kwargs)
    monkeypatch.setattr(pd, "read_csv", tracked_read)
    # The production loader's frozen count checks are separately tested; mock
    # only dataset loading here to train a small synthetic artifact, never on val.
    monkeypatch.setattr(frozen, "load_development_data", lambda: pd.read_csv(dev_path))
    original_fit = Pipeline.fit
    def tracked_fit(self, X, y, **kwargs):
        assert len(X) == n and list(X.columns) == CORE_FEATURES
        assert set(X.index) == set(dev.index)
        pd.testing.assert_frame_equal(X, frozen.validate_input_columns(dev))
        fitted_pipelines.append(self)
        return original_fit(self, X, y, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", tracked_fit)
    output_dir = tmp_path / "artifacts"
    artifact, metadata_path, metadata = frozen.build_frozen_artifact(output_dir)
    assert len(fitted_pipelines) == 1
    assert str(sentinel) not in calls
    assert initial_hashes == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in initial_hashes}
    return dict(artifact=artifact, metadata_path=metadata_path, metadata=metadata, development=dev,
                pipeline=fitted_pipelines[0], thresholds=exact_values, documented=documented,
                output_dir=output_dir, fits=fitted_pipelines)


def test_artifact_reload_equivalence_and_missing_values(built):
    assert built["artifact"].is_file() and built["metadata_path"].is_file()
    pipeline = frozen.load_frozen_model(built["artifact"], built["metadata_path"])
    probe = built["development"][CORE_FEATURES].iloc[:12].copy()
    probe.loc[probe.index[0], :] = np.nan
    original = frozen.predict_probabilities(probe, built["pipeline"])
    reloaded = frozen.predict_probabilities(probe, pipeline)
    np.testing.assert_allclose(original, reloaded, rtol=1e-7, atol=1e-8)
    assert np.isfinite(reloaded).all() and ((reloaded >= 0) & (reloaded <= 1)).all()
    assert len(built["fits"]) == 1
    assert built["metadata"]["serialization"]["reload_equivalent"] is True
    assert built["metadata"]["serialization"]["maximum_absolute_prediction_difference"] <= 1e-8


def test_named_references_exact_and_not_clinical(built):
    X = built["development"].iloc[:8]
    probability = frozen.predict_probabilities(X, built["pipeline"])
    for name, value in built["thresholds"].items():
        assert built["metadata"]["reference_thresholds"][name] == value
        prediction = frozen.predict_at_reference(X, name, artifact_path=built["artifact"], metadata_path=built["metadata_path"])
        np.testing.assert_array_equal(prediction, (probability >= value).astype(int))
    with pytest.raises(ValueError, match="Unknown operating reference"):
        frozen.predict_at_reference(X, "clinical", artifact_path=built["artifact"], metadata_path=built["metadata_path"])


def test_required_columns_names_order_and_extra_fields():
    X = pd.DataFrame({name: [np.nan, 1.] for name in reversed(CORE_FEATURES)})
    X["seqn"], X["diabetes"] = [1, 2], [0, 1]
    original = X.copy(deep=True)
    validated = frozen.validate_input_columns(X)
    assert list(validated) == CORE_FEATURES and validated.isna().iloc[0].all()
    pd.testing.assert_frame_equal(X, original)
    with pytest.raises(ValueError, match="Missing required Core-10 columns.*bmi"):
        frozen.validate_input_columns(X.drop(columns="bmi"))
    with pytest.raises(TypeError, match="DataFrame"):
        frozen.validate_input_columns(np.ones((2, 10)))
    with pytest.raises(ValueError, match="duplicate"):
        frozen.validate_input_columns(pd.concat([X, X[["age"]]], axis=1))
    with pytest.raises(ValueError, match="no participants"):
        frozen.validate_input_columns(X.iloc[:0])
    X.loc[0, "age"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        frozen.validate_input_columns(X)


def test_metadata_provenance_and_frozen_hyperparameters(built):
    metadata = frozen.load_metadata(built["metadata_path"])
    required = {"model_name", "model_version", "project_purpose", "target_definition", "development_cohort",
        "development_n", "positive_n", "negative_n", "core_features", "preprocessing_specification",
        "xgboost_hyperparameters", "post_hoc_calibration", "reference_thresholds", "threshold_statement",
        "development_csv_sha256", "threshold_results_sha256", "artifact_sha256", "python_version",
        "package_versions", "training_started_utc", "training_completed_utc", "random_seed",
        "temporal_validation_status", "temporal_validation_summary", "transformed_feature_names"}
    assert required <= set(metadata)
    assert metadata["development_n"] == 120 and metadata["positive_n"] == 40 and metadata["negative_n"] == 80
    assert metadata["core_features"] == CORE_FEATURES and metadata["input_feature_count"] == 10
    assert metadata["post_hoc_calibration"] == "none" and metadata["class_weight"] == "none"
    for name, value in frozen.FROZEN_XGB_PARAMETERS.items():
        assert built["pipeline"].named_steps["model"].get_params()[name] == value
        assert metadata["xgboost_hyperparameters"][name] == value
    numeric = built["pipeline"].named_steps["preprocessing"].named_transformers_["numeric"]
    assert "scaler" not in numeric.named_steps and numeric.named_steps["imputer"].strategy == "median"
    for name in ["roc_auc", "average_precision", "brier_score", "log_loss", "calibration_slope", "calibration_intercept"]:
        assert metadata["temporal_validation_summary"]["metrics"][name] == built["documented"][name]
    assert metadata["input_sha256_before"] == metadata["input_sha256_after"]
    assert metadata["prior_result_sha256_before"] == metadata["prior_result_sha256_after"]
    assert hashlib.sha256(built["artifact"].read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert "not clinically validated" in metadata["threshold_statement"]
    assert "joblib" in metadata["package_versions"] and "xgboost" in metadata["package_versions"]


def test_frozen_overwrite_guard_integrity_and_version_warning(built, tmp_path):
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        frozen.build_frozen_artifact(built["output_dir"])
    assert len(built["fits"]) == 1
    artifact = tmp_path / "corrupt.joblib"
    shutil.copyfile(built["artifact"], artifact)
    with artifact.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        frozen.load_frozen_model(artifact, built["metadata_path"])
    metadata = built["metadata"].copy()
    metadata["package_versions"] = dict(metadata["package_versions"], joblib="0.0.test")
    altered = tmp_path / "version_metadata.json"
    altered.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
    with pytest.warns(UserWarning, match="environment differs"):
        frozen.load_frozen_model(built["artifact"], altered)

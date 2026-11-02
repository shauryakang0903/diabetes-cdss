import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data.preprocessing import CORE_FEATURES
from src.explainability import nhanes_shap_global as xai
from src.models.nhanes_phase2_model import load_frozen_model, load_metadata


def forbid_fit(*args, **kwargs):
    pytest.fail("Explainability must never fit an estimator or preprocessor")


@pytest.fixture
def frozen_model(monkeypatch):
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier, CalibratedClassifierCV):
        monkeypatch.setattr(cls, "fit", forbid_fit)
    pipeline = load_frozen_model()
    data = pd.read_csv(xai.DEVELOPMENT_PATH, nrows=64)
    return pipeline, data


def test_real_frozen_mapping_and_shap_no_refit(frozen_model):
    pipeline, data = frozen_model
    metadata = load_metadata()
    assert xai.sha256(xai.DEFAULT_ARTIFACT_PATH) == metadata["artifact_sha256"]
    prep = pipeline.named_steps["preprocessing"]
    transformed = prep.transform(data[CORE_FEATURES])
    names = prep.get_feature_names_out().tolist()
    assert transformed.shape == (64, 18) and names == metadata["transformed_feature_names"]
    mapping = xai.feature_mapping(prep)
    pd.testing.assert_frame_equal(mapping, xai.feature_mapping(prep))
    assert mapping.transformed_index.tolist() == list(range(18))
    assert mapping.transformed_feature.tolist() == names
    assert set(mapping.original_core10_feature) == set(CORE_FEATURES)
    assert not mapping.transformed_index.duplicated().any()
    values, base, margin, diagnostics, name = xai.explain_transformed(pipeline.named_steps["model"], transformed, names)
    assert name == "TreeExplainer" and values.shape == (64, 18)
    assert np.isfinite(values).all() and diagnostics["passed"]
    np.testing.assert_allclose(base+values.sum(axis=1), margin, atol=xai.ADDITIVITY_ATOL, rtol=xai.ADDITIVITY_RTOL)
    grouped = xai.aggregate_core10(values, mapping)
    assert grouped.shape == (64, 10)
    np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1), atol=1e-10)
    with pytest.raises(AssertionError):
        xai.verify_additivity(values+1, base, margin)


def test_categorical_signed_cancellation_and_deterministic_ranks(frozen_model):
    pipeline, _ = frozen_model
    mapping = xai.feature_mapping(pipeline.named_steps["preprocessing"])
    values = np.zeros((3, 18))
    sex = mapping.loc[mapping.original_core10_feature == "sex", "transformed_index"].to_numpy()
    values[:, sex[0]], values[:, sex[1]] = 2, -2
    grouped = xai.aggregate_core10(values, mapping)
    assert np.abs(grouped[:, CORE_FEATURES.index("sex")]).mean() == 0
    assert np.abs(values[:, sex]).mean(axis=0).sum() == 4
    summary = xai.global_summary(grouped, CORE_FEATURES, "original_core10_feature")
    pd.testing.assert_frame_equal(summary, xai.global_summary(grouped, CORE_FEATURES, "original_core10_feature"))
    assert summary.original_core10_feature.tolist() == CORE_FEATURES  # fixed tie order
    assert summary["rank"].tolist() == list(range(1, 11))
    bad_mapping = mapping.copy()
    bad_mapping.loc[0, "transformed_index"] = 1
    with pytest.raises(ValueError, match="exactly once"):
        xai.aggregate_core10(values, bad_mapping)


def test_global_outputs_provenance_preserved_files_no_temporal(frozen_model, tmp_path, monkeypatch):
    pipeline, data = frozen_model
    # This synthetic temporal sentinel must not be read or altered by the run.
    temporal = tmp_path / "nhanes_validation_2021_2023.csv"
    temporal.write_text("temporal sentinel: not for explanations", encoding="utf-8")
    prior = tmp_path / "prior_metrics.csv"
    prior.write_text("preserve previous result", encoding="utf-8")
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (xai.DEFAULT_ARTIFACT_PATH, xai.DEFAULT_METADATA_PATH, xai.DEVELOPMENT_PATH, temporal, prior)}
    calls = []
    def load(artifact, metadata):
        calls.append((artifact, metadata))
        assert artifact == xai.DEFAULT_ARTIFACT_PATH and metadata == xai.DEFAULT_METADATA_PATH
        return pipeline
    monkeypatch.setattr(xai, "load_frozen_model", load)
    monkeypatch.setattr(xai, "load_development_data", lambda: data.copy())
    # prior results excludes the synthetic temporal cohort, just as production
    # RESULTS_DIR never includes the processed cohort directory.
    protected = tmp_path / "results"
    protected.mkdir()
    (protected / "prior.csv").write_text("protected", encoding="utf-8")
    monkeypatch.setattr(xai, "RESULTS_DIR", protected)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Unexpected cohort/temporal read"))
    output = protected / "xai"
    transformed, core, mapping, provenance = xai.run_global_shap(output)
    assert len(calls) == 1 and len(transformed) == 18 and len(core) == 10 and len(mapping) == 18
    assert provenance["artifact_hash_verified"] and provenance["fitting_performed"] is False
    assert provenance["temporal_cohort_accessed"] is False and provenance["sample_n"] == 64
    assert provenance["sha256_before"] == provenance["sha256_after"]
    assert provenance["protected_prior_results_sha256_before"] == provenance["protected_prior_results_sha256_after"]
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes}
    arrays = np.load(output / "shap_values_development.npz", allow_pickle=False)
    assert arrays["shap_values"].shape == (64, 18)
    assert arrays["core10_shap_values"].shape == (64, 10)
    assert arrays["seqn"].tolist() == data.seqn.tolist()
    assert arrays["transformed_feature_names"].tolist() == mapping.transformed_feature.tolist()
    np.testing.assert_allclose(arrays["base_values"]+arrays["shap_values"].sum(axis=1), arrays["raw_margin"],
                               atol=xai.ADDITIVITY_ATOL, rtol=xai.ADDITIVITY_RTOL)
    for filename in ("shap_global_transformed_bar.png", "shap_global_beeswarm.png", "shap_global_core10_bar.png"):
        assert (output / "figures" / filename).stat().st_size > 1000
    for filename in ("shap_global_transformed_features.csv", "shap_global_core10.csv", "shap_feature_mapping.csv",
                     "shap_global_provenance.json"):
        assert (output / filename).is_file()
    saved = json.loads((output / "shap_global_provenance.json").read_text(encoding="utf-8"))
    assert saved["model_version"] == "nhanes_phase2_xgb_v1"


def test_additivity_shape_finite_checks():
    with pytest.raises(ValueError, match="Invalid shapes"):
        xai.verify_additivity(np.ones((2, 18)), np.ones(1), np.ones(2))
    with pytest.raises(ValueError, match="nonfinite"):
        xai.verify_additivity(np.full((2, 18), np.nan), np.ones(2), np.ones(2))

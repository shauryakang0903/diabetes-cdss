import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from scripts.demo_nhanes_cdss import synthetic_profiles
from src.data.preprocessing import CORE_FEATURES
from src.decision_support import nhanes_diabetes_cdss as cdss
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, sha256


@pytest.fixture
def service(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Patient inference must not fit or read cohort CSVs")
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    before = {p: sha256(p) for p in (DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH)}
    instance = cdss.NhanesDiabetesCDSS()
    yield instance
    assert before == {p: sha256(p) for p in before}


def test_single_patient_integrity_and_determinism(service):
    for patient in synthetic_profiles().values():
        result = service.predict(patient)
        assert result == service.predict(pd.DataFrame([patient]))
        json.dumps(result, allow_nan=False)
        assert 0 <= result["predicted_probability"] <= 1
        assert result["predicted_probability_percent"] == 100 * result["predicted_probability"]
        rows = result["all_feature_contributions"]
        assert len(rows) == 10 and {r["feature"] for r in rows} == set(CORE_FEATURES)
        assert rows == sorted(rows, key=lambda r: (-r["absolute_shap_magnitude"], CORE_FEATURES.index(r["feature"])))
        for row in rows:
            assert row["original_value"] == patient[row["feature"]]
            assert row["direction"] == cdss.contribution_direction(row["shap_contribution"])
        for sign, direction in (("increasing", "increases_model_score"), ("decreasing", "decreases_model_score")):
            assert result[f"top_{sign}_contributors"] == [r for r in rows if r["direction"] == direction][:3]
        integrity = result["explanation_integrity"]
        assert integrity["margin_additivity"]["passed"] and integrity["probability_check_passed"]
        assert integrity["core10_shap_total"] == pytest.approx(integrity["transformed_shap_total"], abs=1e-10)
        assert sum(r["shap_contribution"] for r in rows) == pytest.approx(integrity["core10_shap_total"])
        assert integrity["reconstructed_margin"] == pytest.approx(integrity["raw_margin"], abs=1e-4)
        assert integrity["reconstructed_probability"] == pytest.approx(result["predicted_probability"], abs=1e-6)
        for name, value in service.metadata["reference_thresholds"].items():
            assert result["thresholds"][name] == {"threshold": value, "at_or_above": result["predicted_probability"] >= value}
        assert result["limitations"] == list(cdss.LIMITATIONS)


def test_all_missing_supported(service):
    result = service.predict(dict.fromkeys(CORE_FEATURES, None))
    assert all(r["was_missing"] and r["original_value"] is None for r in result["all_feature_contributions"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"age": "bad"}, {"bmi": np.inf}, {"sex": "unsupported"}, {"race_ethnicity": "unknown"},
    {"smoking_status": "current"}, {"sex": True}, {"bmi": [22]},
])
def test_invalid_values(service, change):
    patient = next(iter(synthetic_profiles().values())).copy()
    patient.update(change)
    with pytest.raises((ValueError, TypeError)):
        service.predict(patient)


def test_missing_columns_and_multiple_records(service):
    patient = next(iter(synthetic_profiles().values())).copy()
    del patient["age"]
    with pytest.raises(ValueError, match="age"):
        service.predict(patient)
    with pytest.raises(ValueError, match="single patient"):
        service.predict(pd.DataFrame([patient, patient]))
    duplicate = pd.DataFrame([[1] * 11], columns=CORE_FEATURES + ["age"])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        service.predict(duplicate)


def test_directions_and_reference_boundaries(service):
    assert cdss.contribution_direction(0) == "neutral"
    assert cdss.contribution_direction(1e-11) == "neutral"
    assert cdss.contribution_direction(1e-8) == "increases_model_score"
    assert cdss.contribution_direction(-1e-8) == "decreases_model_score"
    refs = service.metadata["reference_thresholds"]
    t80, t90 = refs["sensitivity_80_reference"], refs["sensitivity_90_reference"]
    assert t90 < t80
    assert cdss.reference_comparison(t90/2, refs)[1].startswith("Below both")
    comparisons, text = cdss.reference_comparison(t90, refs)
    assert comparisons["sensitivity_90_reference"]["at_or_above"]
    assert "below the 80%" in text
    assert cdss.reference_comparison(t80, refs)[1].startswith("At or above both")


def test_cached_model_and_explainer(monkeypatch):
    calls = {"load": 0, "explain": 0}
    load, explain = cdss.load_frozen_model, cdss.create_tree_explainer
    def wrapped_load(*args):
        calls["load"] += 1
        return load(*args)
    def wrapped_explain(*args):
        calls["explain"] += 1
        return explain(*args)
    monkeypatch.setattr(cdss, "load_frozen_model", wrapped_load)
    monkeypatch.setattr(cdss, "create_tree_explainer", wrapped_explain)
    instance = cdss.NhanesDiabetesCDSS()
    for patient in synthetic_profiles().values():
        instance.predict(patient)
    assert calls == {"load": 1, "explain": 1}

"""Rule-only boundary tests plus immutable frozen-model synthetic regressions."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from scripts.demo_nhanes_cdss import synthetic_profiles
from src.decision_support import nhanes_recommendations as rules
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.decision_support.prototype_presentation import assess_patient
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, load_metadata, sha256

BASELINE_PATH = Path(__file__).parent / "fixtures/nhanes_cdss_pre_recommendation_baseline.json"


@pytest.fixture(autouse=True)
def immutable_predictor(monkeypatch):
    paths = (DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, rules.CONFIG_PATH, BASELINE_PATH)
    before = {p: sha256(p) for p in paths}
    def forbidden(*args, **kwargs):
        pytest.fail("No model fitting, preprocessing refitting, or cohort CSV access is permitted")
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)
        if hasattr(cls, "fit_transform"):
            monkeypatch.setattr(cls, "fit_transform", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    yield
    assert before == {p: sha256(p) for p in paths}


def baseline_result():
    saved = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return copy.deepcopy(saved["profiles"]["relatively_lower_scoring_example"]["result"])


def rule_only_result(**changes):
    """Change measurements for predicate unit tests, not new model predictions."""
    result = baseline_result()
    values = dict(age=40, sex=1, bmi=22, waist_cm=80, mean_systolic_bp=110,
                  mean_diastolic_bp=70, hdl_mg_dl=60, total_cholesterol_mg_dl=180, smoking_status=0)
    values.update(changes)
    for row in result["all_feature_contributions"]:
        if row["feature"] in values:
            row["original_value"] = values[row["feature"]]
    return result


def triggered(result):
    return set(result["recommendation_provenance"]["triggered_rule_ids"])


def test_exact_frozen_boundaries_and_labels(monkeypatch):
    metadata = load_metadata()
    calls = []
    def stored():
        calls.append(True)
        return metadata
    monkeypatch.setattr(rules, "load_metadata", stored)
    lower = metadata["reference_thresholds"]["sensitivity_90_reference"]
    upper = metadata["reference_thresholds"]["sensitivity_80_reference"]
    cases = [(0., "LOW"), (np.nextafter(lower, -np.inf), "LOW"), (lower, "MODERATE"),
             ((lower+upper)/2, "MODERATE"), (np.nextafter(upper, -np.inf), "MODERATE"),
             (upper, "HIGH"), (np.nextafter(upper, np.inf), "HIGH"), (0.5, "HIGH"), (1., "HIGH")]
    for probability, expected in cases:
        risk = rules.model_derived_risk(probability)
        assert risk["level"] == expected and risk["label"] == f"{expected} model-derived risk"
        assert risk["references"] == {"sensitivity_90_reference": lower, "sensitivity_80_reference": upper}
        assert "not diagnostic categories" in risk["disclaimer"]
        if expected == "MODERATE":
            assert risk["lower_boundary"] == lower and risk["upper_boundary"] == upper
            assert risk["lower_inclusive"] and not risk["upper_inclusive"]
    assert len(calls) == len(cases)
    assert rules.model_derived_risk(metadata["reference_thresholds"]["youden_reference"])["level"] == "MODERATE"
    malformed = copy.deepcopy(metadata)
    malformed["reference_thresholds"]["sensitivity_90_reference"] = upper
    with pytest.raises(ValueError, match="reference90"):
        rules.model_derived_risk(.1, malformed)


@pytest.mark.parametrize("probability", [-.1, 1.1, np.nan, np.inf, True, "0.1"])
def test_invalid_probability(probability):
    with pytest.raises(ValueError, match="probability"):
        rules.model_derived_risk(probability)


def test_central_auditable_rule_definitions():
    config = rules.load_recommendation_rules()
    assert config["config_sha256"] == sha256(rules.CONFIG_PATH)
    assert config["risk_boundaries"] == {"lower_reference": "sensitivity_90_reference", "upper_reference": "sensitivity_80_reference"}
    ids = [r["rule_id"] for r in config["rules"]]
    assert len(ids) == len(set(ids))
    assert len(ids) == 17
    for rule in config["rules"]:
        assert rule["trigger_description"] and rule["rationale"] and rule["source_labels"]
        assert all(source in config["sources"] for source in rule["source_labels"])
        assert "race_ethnicity" not in json.dumps(rule["trigger"])


@pytest.mark.parametrize("smoking,expected", [(0, False), (1, False), (None, False), (2, True)])
def test_current_smoker_rule_only(smoking, expected):
    result = rules.add_recommendation_layer(rule_only_result(smoking_status=smoking))
    assert ("REC_SMOKING_001" in triggered(result)) == expected
    assert set(result["recommendations"]) == set(rules.SECTIONS)


@pytest.mark.parametrize("changes,rule_id,expected", [
    ({"bmi": 24.99}, "REC_WEIGHT_001", False), ({"bmi": 25}, "REC_WEIGHT_001", True),
    ({"mean_systolic_bp": 119.99, "mean_diastolic_bp": 79.99}, "REC_BP_001", False),
    ({"mean_systolic_bp": 120}, "REC_BP_001", True), ({"mean_diastolic_bp": 80}, "REC_BP_001", True),
    ({"sex": 1, "waist_cm": 101.6}, "REC_WAIST_001", False),
    ({"sex": 1, "waist_cm": np.nextafter(101.6, np.inf)}, "REC_WAIST_001", True),
    ({"sex": 2, "waist_cm": 88.9}, "REC_WAIST_001", False),
    ({"sex": 2, "waist_cm": np.nextafter(88.9, np.inf)}, "REC_WAIST_001", True),
    ({"total_cholesterol_mg_dl": 199.99, "hdl_mg_dl": 40}, "REC_LIPID_001", False),
    ({"total_cholesterol_mg_dl": 200}, "REC_LIPID_001", True),
    ({"sex": 1, "hdl_mg_dl": 39.99}, "REC_LIPID_001", True),
    ({"sex": 2, "hdl_mg_dl": 50}, "REC_LIPID_001", False),
    ({"sex": 2, "hdl_mg_dl": 49.99}, "REC_LIPID_001", True),
    ({"age": None, "bmi": 35}, "REC_WEIGHT_001", False),
    ({"age": 19, "bmi": 35}, "REC_WEIGHT_001", False),
])
def test_observed_measurement_thresholds(changes, rule_id, expected):
    result = rules.add_recommendation_layer(rule_only_result(**changes))
    assert (rule_id in triggered(result)) == expected


def test_missing_values_are_not_treated_as_observed_abnormalities():
    result = rule_only_result()
    for row in result["all_feature_contributions"]:
        row["original_value"] = None
    output = rules.add_recommendation_layer(result)
    ids = triggered(output)
    assert {"REC_MISSING_001", "REC_SCOPE_001"} <= ids
    assert not ids & {"REC_WEIGHT_001", "REC_BP_001", "REC_LIPID_001", "REC_WAIST_001", "REC_SMOKING_001"}
    json.dumps(output, allow_nan=False)


def test_shap_signs_and_race_do_not_drive_actions():
    original = rule_only_result(bmi=28, smoking_status=2)
    changed = copy.deepcopy(original)
    for row in changed["all_feature_contributions"]:
        row["shap_contribution"] *= -1
        if row["feature"] == "race_ethnicity":
            row["original_value"] = 7
    first, second = [rules.add_recommendation_layer(r) for r in (original, changed)]
    assert first["recommendations"] == second["recommendations"]
    assert first["model_derived_risk"] == second["model_derived_risk"]
    assert original == rule_only_result(bmi=28, smoking_status=2)  # No mutation.


def test_three_strata_followup_content_and_determinism():
    meta = load_metadata()
    lower, upper = [meta["reference_thresholds"][n] for n in ("sensitivity_90_reference", "sensitivity_80_reference")]
    for probability, level in ((lower/2, "LOW"), ((lower+upper)/2, "MODERATE"), (upper, "HIGH")):
        original = rule_only_result()
        original["predicted_probability"] = probability  # Rule-only boundary simulation.
        output = rules.add_recommendation_layer(original)
        assert output == rules.add_recommendation_layer(original)
        assert f"REC_FOLLOWUP_{level}_001" in triggered(output)
        assert f"REC_MONITOR_{level}_001" in triggered(output)
        assert "REC_SYMPTOMS_001" in triggered(output)
        assert set(output["recommendations"]) == set(rules.SECTIONS)
        assert "https://" not in json.dumps(output, allow_nan=False)
        for section in rules.SECTIONS:
            for row in output["recommendations"][section]:
                assert row["source_ids"] and row["source_labels"] and row["trigger_description"]
        assert output["recommendation_provenance"]["clinician_review_required"]


def test_identical_pre_framework_predictions_shap_and_ui_delegation():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["artifact_sha256"] == sha256(DEFAULT_ARTIFACT_PATH)
    service = NhanesDiabetesCDSS()
    expected = ["LOW", "LOW", "HIGH"]
    for (name, patient), level in zip(synthetic_profiles().items(), expected):
        result = assess_patient(patient, service)
        old = baseline["profiles"][name]["result"]
        assert all(result[key] == value for key, value in old.items())
        assert result["model_derived_risk"]["level"] == level
        json.dumps(result, allow_nan=False)
        class Spy:
            def predict(self, record):
                assert record is patient
                return result
        assert assess_patient(patient, Spy()) is result


def test_scenario_outputs_are_auditable(tmp_path):
    from scripts.run_nhanes_risk_recommendation_scenarios import run
    frame, provenance = run(tmp_path)
    assert len(frame) == 3 and frame.risk_stratum.tolist() == ["LOW", "LOW", "HIGH"]
    assert {"scenario", "probability", "risk_stratum", "triggered_rule_ids", "clinical_followup_count", "monitoring_count", "lifestyle_count", "feature_aware_count"} == set(frame.columns)
    assert all(len(json.loads(ids)) > 0 for ids in frame.triggered_rule_ids)
    assert frame.feature_aware_count.tolist() == [0, 2, 5]
    assert provenance["sha256_before"] == provenance["sha256_after"]
    assert provenance["frozen_predictions_identical"] and provenance["shap_outputs_identical"]
    assert not provenance["model_retrained"] and not provenance["temporal_cohort_used"]
    assert not provenance["thresholds_optimized"] and not provenance["cohort_data_loaded"]
    assert len(list(tmp_path.iterdir())) == 2

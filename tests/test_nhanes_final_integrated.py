"""Final integrated validation using synthetic inputs and saved evidence only."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from streamlit.testing.v1 import AppTest
from xgboost import XGBClassifier

from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.decision_support.nhanes_recommendations import SECTIONS
from src.decision_support.prototype_presentation import assess_patient, local_explanation_figure, probability_text
from src.validation import nhanes_cdss_integrated as final


@pytest.fixture(autouse=True)
def protect_system(monkeypatch):
    before = final.snapshot_inputs()
    def forbidden(*args, **kwargs):
        pytest.fail("Final validation must not fit the frozen system")
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)
        if hasattr(cls, "fit_transform"):
            monkeypatch.setattr(cls, "fit_transform", forbidden)
    read = pd.read_csv
    allowed = {path.resolve() for path in final.HISTORY.values() if path.suffix == ".csv"}
    def saved_results_only(path, *args, **kwargs):
        assert Path(path).resolve() in allowed, f"Unexpected cohort/result read: {path}"
        return read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", saved_results_only)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    assert before == final.snapshot_inputs()


def test_integrated_scenarios_preserve_all_layers():
    frame, results = final.validate_scenarios(repetitions=3)
    assert frame.model_derived_stratum.tolist() == ["LOW", "LOW", "HIGH", "MODERATE"]
    assert frame.backend_deterministic.all() and frame.presentation_deterministic.all()
    assert frame.frozen_probability_identical.all() and frame.recommendation_layer_preserves_output.all()
    assert frame.legacy_output_identical.iloc[:3].all()
    assert frame.feature_aware_count.tolist() == [0, 2, 5, 2]
    high = results["higher_scoring_example"]
    ids = set(high["recommendation_provenance"]["triggered_rule_ids"])
    assert {"REC_SMOKING_001", "REC_WEIGHT_001", "REC_BP_001", "REC_WAIST_001", "REC_LIPID_001"} <= ids
    assert "REC_SMOKING_001" not in results["intermediate_scoring_example"]["recommendation_provenance"]["triggered_rule_ids"]
    for result in results.values():
        assert set(result["recommendations"]) == set(SECTIONS)
        assert result["limitations"] and result["model_derived_risk"]["disclaimer"]
        json.dumps(final.presentation_snapshot(result), allow_nan=False)


def test_model_integrity_saved_metrics_and_boundaries():
    service = NhanesDiabetesCDSS()
    assert final.verify_integrity(service)["passed"]
    assert all(case["passed"] for case in final.boundary_checks(service.metadata))
    summary = final.historical_summary()
    assert not summary["predictive_experiments_rerun"] and not summary["temporal_data_reopened"]
    assert summary["development"]["roc_auc"]["mean"] > summary["temporal"]["roc_auc"]
    assert summary["temporal"]["missingness_groups"][0]["roc_auc"] > summary["temporal"]["missingness_groups"][1]["roc_auc"]
    assert 0 < summary["threshold_transport"]["0.80"]["sensitivity"] < summary["threshold_transport"]["0.90"]["sensitivity"] < 1
    assert len(summary["explainability"]["shap_lime_agreement"]) == 3
    json.dumps(summary, allow_nan=False)


def test_input_errors_through_existing_ui_integration_boundary():
    service = NhanesDiabetesCDSS()
    patient = next(iter(final.validation_scenarios().values())).copy()
    del patient["age"]
    with pytest.raises(ValueError, match="Missing required.*age"):
        assess_patient(patient, service)
    patient["age"], patient["sex"] = 40, 'invalid'
    with pytest.raises(ValueError, match="numeric"):
        assess_patient(patient, service)


@pytest.mark.parametrize("text", ["You have diabetes.", "You do not have diabetes.",
    "You are diagnosed with diabetes.", "These are clinically validated thresholds.",
    "A guaranteed safe result.", "Critical diabetes.", "Emergency score.",
    "Take metformin 500 mg twice daily.", "Use insulin 10 units.",
    "This is not a diagnosis. These are clinically validated thresholds.",
    "They do not diagnose disease, but prescribe medication."])
def test_language_audit_detects_unsupported_claims(text):
    assert final.language_findings(text)[0]


@pytest.mark.parametrize("text", ["Physician-diagnosed diabetes status, not prospective incidence.",
    "These are not clinically validated diagnostic thresholds.",
    "They are not diagnostic categories or clinically validated risk thresholds.",
    "HDL cholesterol: 48 mg/dL.", "Model output is not a diagnosis.",
    "No medications are prescribed.",
    "They do not diagnose disease, replace a treatment plan, or prescribe medication."])
def test_language_audit_preserves_appropriate_context(text):
    assert not final.language_findings(text)[0]


def test_traceability_is_evidence_based_and_not_uniformly_complete():
    table = final.objective_traceability()
    assert len(table) == 21 and table.requirement_id.is_unique
    assert set(table.synopsis_objective) == {"OBJ1", "OBJ2", "OBJ3", "OBJ4"}
    assert set(table.status) <= {"COMPLETE", "PARTIAL", "NOT IMPLEMENTED"}
    assert table.loc[table.requirement_id == "OBJ2_LIME", "status"].iloc[0] == "PARTIAL"
    assert table.loc[table.requirement_id == "OBJ1_PRED", "status"].iloc[0] == "PARTIAL"
    assert (final.objective_traceability(runtime_ok=False).status != "COMPLETE").all()


def test_pytest_report_parsing(tmp_path):
    path = tmp_path/"example.xml"
    path.write_text('<testsuites><testsuite><testcase name="pass"/><testcase name="skip"><skipped/></testcase><testcase name="fail"><failure/></testcase></testsuite></testsuites>', encoding="utf-8")
    result = final.read_pytest_report(path)
    assert (result["test_count"], result["passed"], result["skipped"], result["failures"]) == (3, 1, 1, 1)


def test_final_streamlit_smoke(monkeypatch):
    calls = []
    predict = NhanesDiabetesCDSS.predict
    def spy(self, patient):
        calls.append(patient.copy())
        return predict(self, patient)
    monkeypatch.setattr(NhanesDiabetesCDSS, "predict", spy)
    app = AppTest.from_file(str(final.ROOT/"app.py"), default_timeout=60).run()
    assert not app.exception and not app.error and not calls
    assert len(app.number_input) == 7 and len(app.selectbox) == 4
    captured = set()
    for kind in ("title", "caption", "markdown", "warning", "info", "subheader"):
        captured.update(str(item.value) for item in getattr(app, kind))
    observed = {}
    for name, patient in final.validation_scenarios().items():
        if name == "moderate_stratum_software_demonstration":
            app.selectbox[0].set_value("intermediate_scoring_example").run()
            app.number_input[0].set_value(float(patient["age"]))
        else:
            app.selectbox[0].set_value(name).run()
        previous = len(calls)
        app.button[0].click().run()
        assert not app.exception and not app.error
        assert len(calls) == previous+1 and calls[-1] == patient
        result = app.session_state["assessment"]
        assert app.metric[0].value == probability_text(result)
        markdown = "\n".join(item.value for item in app.markdown)
        assert result["model_derived_risk"]["label"] in markdown
        assert [tab.label for tab in app.tabs] == ["Clinical follow-up", "Monitoring", "Lifestyle", "Patient-specific considerations"]
        assert len(app.dataframe[1].value) == 10
        assert "Why the model predicted this — SHAP" in [s.value for s in app.subheader]
        for section in SECTIONS:
            for rule in result["recommendations"][section]:
                assert rule["text"] in markdown
        assert "Model and validation information" in [e.label for e in app.expander]
        fig = local_explanation_figure(result)
        FigureCanvasAgg(fig).draw()
        np.testing.assert_allclose([patch.get_width() for patch in fig.axes[0].patches],
                                   [r["shap_contribution"] for r in reversed(result["all_feature_contributions"])], atol=0, rtol=0)
        for kind in ("title", "caption", "markdown", "warning", "info", "subheader"):
            captured.update(str(item.value) for item in getattr(app, kind))
        observed[name] = dict(probability=result["predicted_probability"], displayed_probability=app.metric[0].value,
                              risk_label=result["model_derived_risk"]["label"], backend_called=True,
                              shap_rendered=True, recommendations_rendered=True)
    report = dict(passed=True, run_id=os.environ.get("NHANES_FINAL_VALIDATION_RUN_ID"),
                  method="Streamlit AppTest with actual frozen backend and Matplotlib canvas rendering",
                  checks={key: True for key in ("input_form_loaded", "profile_loading_works", "submit_calls_backend",
                    "probability_displays", "all_three_strata_display", "shap_section_renders", "recommendation_tabs_render",
                    "model_information_renders", "disclaimers_render", "no_prediction_on_initial_load")},
                  scenarios=observed, rendered_text=sorted(captured))
    service = NhanesDiabetesCDSS()
    # All 17 configured texts are audited, including rules not triggered by these profiles.
    audit = final.clinical_language_audit({}, service.recommendation_rules, report)
    assert not (audit.status == "FLAGGED").any(), audit[audit.status == "FLAGGED"].to_string()
    path = os.environ.get("NHANES_FINAL_SMOKE_REPORT")
    if path:
        Path(path).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

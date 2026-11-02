"""Small presentation and Streamlit smoke tests using synthetic records only."""

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

from scripts.demo_nhanes_cdss import synthetic_profiles
from src.data.preprocessing import CORE_FEATURES
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.decision_support import prototype_presentation as view
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, sha256

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def inference_only(monkeypatch):
    before = {p: sha256(p) for p in (DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH)}
    def forbidden(*args, **kwargs):
        pytest.fail("The prototype must never fit or read cohort CSVs")
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    assert before == {p: sha256(p) for p in before}


def test_assessment_delegates_without_changing_backend_output():
    patient = next(iter(synthetic_profiles().values()))
    sentinel = {"predicted_probability_percent": 18.4, "reference_interpretation": "unchanged"}
    class Spy:
        def predict(self, record):
            assert record is patient
            return sentinel
    result = view.assess_patient(patient, Spy())
    assert result is sentinel
    assert view.probability_text(result) == "18.40%"
    assert result["reference_interpretation"] == "unchanged"


def test_real_synthetic_results_tables_and_chart():
    service = NhanesDiabetesCDSS()
    for patient in synthetic_profiles().values():
        result = view.assess_patient(patient, service)
        assert view.probability_text(result) == f"{100 * result['predicted_probability']:.2f}%"
        table = view.explanation_table(result)
        assert len(table) == 10 and set(table["Core-10 feature"]) == set(CORE_FEATURES)
        np.testing.assert_array_equal(table["SHAP contribution (log-odds)"],
                                      [r["shap_contribution"] for r in result["all_feature_contributions"]])
        refs = view.reference_table(result)
        assert refs.Threshold.tolist() == [r["threshold"] for r in result["thresholds"].values()]
        fig = view.local_explanation_figure(result)
        FigureCanvasAgg(fig).draw()  # Exercise actual chart rendering.
        bars = fig.axes[0].patches
        assert len(bars) == 10
        assert "log-odds" in fig.axes[0].get_xlabel()
        np.testing.assert_array_equal([b.get_width() for b in bars],
                                      [r["shap_contribution"] for r in reversed(result["all_feature_contributions"])])
        assert len({b.get_facecolor() for b in bars}) >= 2


def test_readable_category_codes_and_missing_display():
    assert list(view.CATEGORY_LABELS["sex"]) == [1, 2]
    assert list(view.CATEGORY_LABELS["race_ethnicity"]) == [1, 2, 3, 4, 6, 7]
    assert list(view.CATEGORY_LABELS["smoking_status"]) == [0, 1, 2]
    assert view.display_value({"feature": "smoking_status", "original_value": 1}) == "Former smoker"
    assert "Unavailable" in view.display_value({"feature": "age", "original_value": None})


def test_streamlit_submission_profiles_and_rendering(monkeypatch):
    calls = []
    predict = NhanesDiabetesCDSS.predict
    def spy(self, patient):
        calls.append(patient.copy())
        return predict(self, patient)
    monkeypatch.setattr(NhanesDiabetesCDSS, "predict", spy)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception and not app.error and not calls
    assert len(app.number_input) == 7 and len(app.selectbox) == 4
    app.number_input[0].set_value(49.0).run()
    assert not calls  # Editing does not assess.
    for index, (name, patient) in enumerate(synthetic_profiles().items(), 1):
        app.selectbox[0].set_value(name).run()
        assert len(calls) == index - 1
        assert "assessment" not in app.session_state
        app.button[0].click().run()
        assert not app.exception and not app.error
        assert calls[-1] == patient and len(calls) == index
        result = app.session_state["assessment"]
        assert app.metric[0].value == view.probability_text(result)
        assert result["reference_interpretation"] in [item.value for item in app.info]
        assert result["model_derived_risk"]["label"] in " ".join(item.value for item in app.markdown)
        assert [tab.label for tab in app.tabs] == ["Clinical follow-up", "Monitoring", "Lifestyle", "Patient-specific considerations"]
        assert "Model-derived decision-support category — not a diagnosis." in [item.value for item in app.markdown]
        assert len(app.dataframe) == 2
        assert len(app.dataframe[1].value) == 10


def test_input_error_is_readable_and_clears_result(monkeypatch):
    def invalid(self, patient):
        raise ValueError("Missing required Core-10 field: age")
    monkeypatch.setattr(NhanesDiabetesCDSS, "predict", invalid)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception and not app.metric
    assert "age" in app.error[0].value
    assert "assessment" not in app.session_state


def test_artifact_load_error_is_readable(monkeypatch):
    def broken(self):
        raise OSError("Internal file details must not appear in UI")
    monkeypatch.setattr(NhanesDiabetesCDSS, "__init__", broken)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception and not app.metric
    assert "could not be loaded" in app.error[0].value
    assert "Internal file details" not in app.error[0].value


def test_unexpected_prediction_error_is_readable(monkeypatch):
    def broken(self, patient):
        raise AssertionError("Private integrity details")
    monkeypatch.setattr(NhanesDiabetesCDSS, "predict", broken)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception and not app.metric
    assert "integrity check" in app.error[0].value
    assert "Private integrity details" not in app.error[0].value

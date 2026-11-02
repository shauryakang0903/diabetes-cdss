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
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_preprocessing_comparison import DEVELOPMENT_PATH
from src.explainability import nhanes_lime_local as lime
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, sha256


@pytest.fixture
def service(monkeypatch):
    paths = (DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, DEVELOPMENT_PATH)
    before = {p: sha256(p) for p in paths}
    def forbid(*args, **kwargs):
        pytest.fail("LIME must not fit the frozen predictor or its preprocessing")
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier):
        monkeypatch.setattr(cls, "fit", forbid)
        if hasattr(cls, "fit_transform"):
            monkeypatch.setattr(cls, "fit_transform", forbid)
    read = pd.read_csv
    calls = []
    def development_only(path, *args, **kwargs):
        assert Path(path).resolve() == DEVELOPMENT_PATH.resolve(), "Unexpected cohort access"
        calls.append(Path(path))
        return read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", development_only)
    instance = lime.NhanesLimeLocal()
    assert calls == [DEVELOPMENT_PATH]
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Explanation must not reread cohorts"))
    yield instance
    assert before == {p: sha256(p) for p in paths}


def test_real_frozen_predictions_and_deterministic_ranks(service, monkeypatch):
    patient = synthetic_profiles()["higher_scoring_example"]
    original = service.backend.predict(patient)
    calls = []
    predictor = lime.predict_probabilities
    def spy(frame, *, pipeline):
        assert pipeline is service.backend.pipeline
        assert frame.columns.tolist() == CORE_FEATURES and len(frame) == 5000
        for feature, allowed in service.category_codes.items():
            assert frame[feature].isin(allowed).all()
        calls.append(len(frame))
        return predictor(frame, pipeline=pipeline)
    monkeypatch.setattr(lime, "predict_probabilities", spy)
    first = service.explain_patient(patient)
    second = service.explain_patient(patient)
    assert first == second and calls == [5000, 5000]
    json.dumps(first, allow_nan=False)
    assert first["predicted_probability"] == original["predicted_probability"]
    assert service.backend.predict(patient) == original
    rows = first["local_feature_weights"]
    assert len(rows) == 10 and {r["feature"] for r in rows} == set(CORE_FEATURES)
    assert [r["rank"] for r in rows] == list(range(1, 11))
    assert rows == sorted(rows, key=lambda r: (-abs(r["lime_weight"]), CORE_FEATURES.index(r["feature"])))
    for row in rows:
        assert row["direction"] == lime.weight_direction(row["lime_weight"])
    assert first["local_intercept"] + sum(r["lime_weight"] for r in rows) == pytest.approx(first["surrogate_prediction"])
    assert np.isfinite(first["local_weighted_r2"])


def test_category_roundtrip_and_missing_predictors(service):
    frame = pd.DataFrame(list(synthetic_profiles().values()))
    frame["race_ethnicity"] = [1., 6., 7.]
    encoded = service._encode(frame)
    assert set(encoded[:, CORE_FEATURES.index("race_ethnicity")]) == {0., 4., 5.}
    pd.testing.assert_frame_equal(service._decode(encoded), frame.astype(float), check_dtype=False)
    invalid = encoded.copy()
    invalid[0, CORE_FEATURES.index("sex")] = 0.5
    with pytest.raises(ValueError, match="categorical index"):
        service.predict_lime_samples(invalid)
    result = service.explain_patient(dict.fromkeys(CORE_FEATURES, None), num_samples=500)
    assert all(r["original_value"] is None for r in result["local_feature_weights"])
    assert 0 <= result["predicted_probability"] <= 1


def fake_explanations():
    shap = {"all_feature_contributions": [dict(feature=c, shap_contribution=10-i) for i, c in enumerate(CORE_FEATURES)]}
    # SHAP top3 A,B,C; LIME top3 A,C,D => overlap 2/3, one direction match.
    order = [0, 2, 3, 5, 6, 1, 4, 7, 8, 9]
    rows = [dict(feature=CORE_FEATURES[i], lime_weight=(10-rank)*(-1 if i == 2 else 1)) for rank, i in enumerate(order)]
    return shap, {"local_feature_weights": rows}


def test_overlap_direction_agreement_and_order_independence():
    shap, explanation = fake_explanations()
    comparison = lime.compare_shap_lime(shap, explanation)
    assert comparison["top3_overlap"] == pytest.approx(2/3)
    assert comparison["top3_direction_agreement"] == 0.5
    assert comparison["top5_overlap"] == 3/5
    assert comparison["top5_direction_agreement"] == pytest.approx(2/3)
    assert comparison["direction_agreement_all_features"] == 0.9
    shap["all_feature_contributions"].reverse()
    explanation["local_feature_weights"].reverse()
    assert comparison == lime.compare_shap_lime(shap, explanation)
    with pytest.raises(ValueError, match="ten unique"):
        lime.compare_shap_lime({"all_feature_contributions": []}, explanation)


def test_disjoint_overlap_and_neutral_ties():
    shap, explanation = fake_explanations()
    for index, row in enumerate(explanation["local_feature_weights"]):
        row["lime_weight"] = CORE_FEATURES.index(row["feature"]) + 1
    result = lime.compare_shap_lime(shap, explanation)
    assert result["top3_overlap"] == 0
    assert result["top3_direction_agreement"] is None
    assert lime.weight_direction(0) == lime.weight_direction(1e-11) == "neutral"


def test_stability_counts_and_pairwise_definition():
    _, explanation = fake_explanations()
    a = dict(random_seed=42, **copy.deepcopy(explanation))
    b = dict(random_seed=43, **copy.deepcopy(explanation))
    rows = lime.summarize_stability([a, b])
    assert len(rows) == 10
    assert sum(r["top3_selection_count"] for r in rows) == 6
    assert sum(r["top5_selection_count"] for r in rows) == 10
    assert all(r["mean_pairwise_top3_overlap"] == 1 for r in rows)
    with pytest.raises(ValueError, match="distinct"):
        lime.summarize_stability([a, a])


def test_runner_outputs_and_provenance(service, monkeypatch, tmp_path):
    from scripts import run_nhanes_lime_local as runner
    # Small test-only perturbation count; production run is fixed at 5,000.
    monkeypatch.setattr(runner, "NhanesLimeLocal", lambda: service)
    monkeypatch.setattr(runner, "NUM_SAMPLES", 300)
    monkeypatch.setattr(runner, "STABILITY_SEEDS", (42, 43))
    prior = tmp_path / "shap_global_core10.csv"
    prior.write_text("do not overwrite", encoding="utf-8")
    before = sha256(prior)
    explanations, agreement, stability, provenance = runner.run(tmp_path)
    assert len(explanations) == len(stability) == 30 and len(agreement) == 3
    assert {"lime_weight", "lime_term", "rank", "feature", "local_weighted_r2", "predicted_probability"} <= set(explanations)
    assert {"top3_overlap", "top5_overlap", "top3_direction_agreement", "top5_direction_agreement"} <= set(agreement)
    assert set(p.name for p in tmp_path.iterdir()) == runner.RESULT_NAMES | {prior.name}
    assert before == sha256(prior)
    assert provenance["sha256_before"] == provenance["sha256_after"]
    assert not provenance["frozen_model_refit"] and not provenance["frozen_preprocessing_refit"]
    assert not provenance["temporal_cohort_accessed"] and provenance["lime_surrogate_fitting"]
    json.loads((tmp_path / "lime_provenance.json").read_text(encoding="utf-8"))

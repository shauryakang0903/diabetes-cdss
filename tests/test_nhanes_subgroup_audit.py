import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_subgroup_audit as audit
from src.models.nhanes_phase2_model import load_metadata, sha256


@pytest.fixture(autouse=True)
def no_frozen_fitting(monkeypatch):
    paths = (audit.DEFAULT_ARTIFACT_PATH, audit.DEFAULT_METADATA_PATH)
    hashes = {p: sha256(p) for p in paths}
    def forbidden(*args, **kwargs):
        pytest.fail("Subgroup auditing must not fit any predictive estimator/preprocessor")
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)
        if hasattr(cls, "fit_transform"):
            monkeypatch.setattr(cls, "fit_transform", forbidden)
    yield
    assert hashes == {p: sha256(p) for p in paths}


def synthetic_frame(n=180):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({c: rng.normal(50, 5, n) for c in CORE_FEATURES})
    frame["age"] = np.tile([20., 39., 40., 59., 60., 80.], n//6)
    frame["bmi"] = np.tile([24., 25., 29., 30., 35., np.nan], n//6)
    frame["sex"] = np.tile([1., 2.], n//2)
    frame["race_ethnicity"] = np.tile([1., 2., 3., 4., 6., 7.], n//6)
    frame["smoking_status"] = 0.
    frame["diabetes"] = np.tile([0, 1, 0, 1, 1, 0], n//6)
    frame["seqn"] = np.arange(n)+1
    frame["nhanes_cycle"] = "2011_2012"
    return frame


def test_age_bmi_boundaries_and_completeness():
    assert audit.age_groups([20, 39.999, 40, 59.999, 60, 80, np.nan]).tolist() == ["20–39", "20–39", "40–59", "40–59", "60+", "60+", "Missing age"]
    with pytest.raises(ValueError, match="adult"):
        audit.age_groups([19])
    assert audit.bmi_groups([24.999, 25, 29.999, 30, np.nan]).tolist() == ["<25", "25–<30", "25–<30", ">=30", "Missing BMI"]
    frame = synthetic_frame()
    groups = audit.completeness_groups(frame)
    assert (groups == ">=1 missing Core-10 input").sum() == 30
    frame.loc[0, "sex"] = np.nan
    assert audit.completeness_groups(frame).iloc[0] == ">=1 missing Core-10 input"
    with pytest.raises(ValueError, match="age"):
        audit.completeness_groups(frame.drop(columns="age"))


def test_existing_mapping_and_family_partitions():
    frame = synthetic_frame()
    definitions = audit.subgroup_definitions(frame)
    assert [label for _, label, _ in definitions["race_ethnicity"]] == list(audit.CATEGORY_LABELS["race_ethnicity"].values())
    assert audit.CATEGORY_LABELS["race_ethnicity"][6] == "Non-Hispanic Asian"
    assert audit.CATEGORY_LABELS["race_ethnicity"][7] == "Other race / multiracial"
    assert audit.FAMILY_TYPES["bmi"] == "clinical" and audit.FAMILY_TYPES["completeness"] == "robustness"
    for family, groups in definitions.items():
        assert np.all(np.column_stack([mask for _, _, mask in groups]).sum(axis=1) == 1)
    frame.loc[0, "race_ethnicity"] = 5
    with pytest.raises(ValueError, match="race_ethnicity"):
        audit.subgroup_definitions(frame)


def test_fixed_threshold_counts_and_undefined_metrics():
    threshold = load_metadata()["reference_thresholds"]["sensitivity_80_reference"]
    result = audit.point_metrics([1, 1, 0, 0], [.8, .01, threshold, .02], threshold)
    assert [result[c] for c in ("tp", "fp", "tn", "fn")] == [1, 1, 1, 1]
    for key in ("sensitivity", "specificity", "precision", "npv", "f1"):
        assert result[key] == .5
    result = audit.point_metrics([0, 0], [.1, .1], threshold)
    assert np.isnan(result["roc_auc"]) and np.isnan(result["average_precision"])
    assert np.isnan(result["sensitivity"]) and np.isnan(result["precision"])
    empty = audit.point_metrics([], [], threshold)
    assert empty["n"] == 0 and np.isnan(empty["prevalence"])
    with pytest.raises(ValueError, match="Probabilities"):
        audit.point_metrics([0, 1], [-.1, .5], threshold)


def test_tie_aware_bootstrap_metrics_equal_sklearn_resampling():
    y = np.tile([0, 1, 1, 0, 1, 0], 20)
    p = np.tile([.1, .1, .4, .6, .9, .9], 20)
    order = np.argsort(-p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order]) != 0)+1]
    rng = np.random.default_rng(42)
    for _ in range(20):
        indices = rng.integers(len(y), size=len(y))
        counts = np.bincount(indices, minlength=len(y))
        auc, ap = audit._weighted_rank_metrics(y[order], counts[order], starts)
        assert auc == pytest.approx(roc_auc_score(y[indices], p[indices]), abs=1e-14)
        assert ap == pytest.approx(average_precision_score(y[indices], p[indices]), abs=1e-14)
    counts = np.where(y == 1, 1, 0)
    assert all(np.isnan(v) for v in audit._weighted_rank_metrics(y[order], counts[order], starts))


def test_bootstrap_reproducibility_and_percentile_reference():
    y = np.tile([0, 1], 60)
    p = np.linspace(.01, .99, len(y))
    t = load_metadata()["reference_thresholds"]["sensitivity_80_reference"]
    first = audit.bootstrap_intervals(y, p, t, seed=42, replicates=40)
    assert first == audit.bootstrap_intervals(y, p, t, seed=42, replicates=40)
    rng, metrics = np.random.default_rng(42), []
    for _ in range(40):
        indices = rng.integers(len(y), size=len(y))
        metrics.append(audit.point_metrics(y[indices], p[indices], t))
    for key in audit.CI_METRICS:
        expected = np.quantile([r[key] for r in metrics], [.025, .975], method="linear")
        assert first[key+"_ci_lower"] == pytest.approx(expected[0])
        assert first[key+"_ci_upper"] == pytest.approx(expected[1])
        assert first[key+"_bootstrap_valid"] == 40
    assert audit.subgroup_seed(42, "development", "sex", "1") == audit.subgroup_seed(42, "development", "sex", "1")
    assert audit.subgroup_seed(42, "temporal", "sex", "1") != audit.subgroup_seed(42, "development", "sex", "1")


def test_unstable_groups_remain_but_cis_suppressed():
    result = audit.bootstrap_intervals([0]*90+[1]*10, [.1]*100, .14, replicates=20)
    assert result["estimates_unstable"] and result["bootstrap_computed"] == 0
    assert "positive count < 30" in result["instability_reason"]
    assert all(np.isnan(result[m+"_ci_lower"]) for m in audit.CI_METRICS)
    frame = synthetic_frame()
    metrics = audit.analyze_cohort(frame, np.linspace(.01, .99, len(frame)), load_metadata(), cohort="temporal", replicates=10)
    assert set(metrics.group_label) >= set(audit.CATEGORY_LABELS["race_ethnicity"].values())
    assert metrics.estimates_unstable.any() and set(metrics.analysis_label) == {audit.TEMPORAL_LABEL}
    assert (metrics.threshold == load_metadata()["reference_thresholds"]["sensitivity_80_reference"]).all()
    assert {"n", "positive_count", "negative_count", "prevalence", "roc_auc", "average_precision", "brier_score", "sensitivity", "specificity", "precision", "npv", "f1"} <= set(metrics)


def test_one_class_bootstrap_draws_use_metric_specific_denominators(monkeypatch):
    # Test-only override to exercise invalid draws normally prevented by the
    # production small-group policy; production cutoffs remain unchanged.
    for name in ("MIN_N", "MIN_POSITIVES", "MIN_NEGATIVES"):
        monkeypatch.setattr(audit, name, 0)
    result = audit.bootstrap_intervals([0, 1], [.1, .9], .14, seed=42, replicates=100)
    assert result["bootstrap_one_class_draws"] > 0
    assert result["roc_auc_bootstrap_valid"] == result["average_precision_bootstrap_valid"]
    assert result["roc_auc_bootstrap_valid"] < result["brier_score_bootstrap_valid"] == 100
    assert result["ci_status"] == "INSUFFICIENT_VALID_BOOTSTRAP_DRAWS"
    assert np.isnan(result["roc_auc_ci_lower"])


def test_saved_oof_reused_and_temporal_frozen_inference_only(monkeypatch):
    load, predict = audit.load_frozen_model, audit.predict_probabilities
    calls, loaded = [], []
    def loader(artifact, metadata):
        assert artifact == audit.DEFAULT_ARTIFACT_PATH and metadata == audit.DEFAULT_METADATA_PATH
        pipeline = load(artifact, metadata)
        loaded.append(pipeline)
        return pipeline
    def predictor(data, *, pipeline):
        assert pipeline is loaded[0]
        assert len(data) == 7536 and set(data.nhanes_cycle) == {"2021_2023"}
        calls.append(len(data))
        return predict(data, pipeline=pipeline)
    monkeypatch.setattr(audit, "load_frozen_model", loader)
    monkeypatch.setattr(audit, "predict_probabilities", predictor)
    dev, dp, temporal, tp, metadata = audit.load_prediction_inputs()
    assert len(dev) == len(dp) == 25580 and len(temporal) == len(tp) == 7536
    assert calls == [7536] and len(loaded) == 1
    assert not set(dev.seqn) & set(temporal.seqn)


def test_disparity_summary_does_not_drop_unstable_groups():
    frame = synthetic_frame()
    metrics = audit.analyze_cohort(frame, np.linspace(.01, .99, len(frame)), load_metadata(), cohort="development", replicates=10)
    ranges = audit.disparity_summary(metrics)
    assert len(ranges) == 20
    for row in ranges.itertuples():
        values = metrics.loc[metrics.subgroup_family == row.subgroup_family, row.metric].dropna()
        if len(values) >= 2:
            assert row.performance_range == pytest.approx(values.max()-values.min())
        else:
            assert np.isnan(row.performance_range)
    assert "stable_groups_performance_range" in ranges


def test_runner_output_schema_provenance_and_plots(tmp_path, monkeypatch):
    folder = tmp_path/"results"
    folder.mkdir()
    out = folder/"subgroup_audit"
    out.mkdir()
    (out/"pytest_full_report.xml").write_text("independent test artifact", encoding="utf-8")
    data = synthetic_frame()
    temporal = data.copy()
    temporal["seqn"] += 1000
    p = np.linspace(.01, .99, len(data))
    meta = load_metadata()
    monkeypatch.setattr(audit, "load_prediction_inputs", lambda: (data, p, temporal, p.copy(), meta))
    monkeypatch.setattr(audit, "AUDIT_DIR", out)
    monkeypatch.setattr(audit, "OUTPUT_DIR", folder)
    overall = audit.point_metrics(temporal.diabetes, p, meta["reference_thresholds"]["sensitivity_80_reference"])
    pd.DataFrame([dict(model="XGBoost", **overall)]).to_csv(folder/"temporal_validation_metrics.csv", index=False)
    dev, val, dr, vr, summary = audit.run_audit(out, replicates=10)
    assert summary["sha256_before"] == summary["sha256_after"]
    assert summary["temporal_status"] == audit.TEMPORAL_LABEL and summary["prior_temporal_evaluation_acknowledged"]
    assert not summary["model_fitting"] and not summary["threshold_optimization"]
    assert (out/"development_subgroup_metrics.csv").is_file()
    assert (out/"temporal_subgroup_metrics.csv").is_file()
    assert (out/"development_disparity_summary.csv").is_file()
    assert (out/"temporal_disparity_summary.csv").is_file()
    assert len(list((out/"figures").glob("*.png"))) == 4
    assert len(list((out/"figures").glob("*.pdf"))) == 4
    assert "pytest_full_report.xml" not in summary["output_sha256"]
    assert (out/"pytest_full_report.xml").read_text(encoding="utf-8") == "independent test artifact"
    json.loads((out/"subgroup_audit_summary.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="outputs must stay"):
        audit.run_audit(tmp_path/"elsewhere", replicates=10)

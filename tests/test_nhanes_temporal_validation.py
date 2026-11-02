import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.data.load_nhanes import DEVELOPMENT_CYCLES
from src.data.nhanes_model_preprocessing import NUMERIC_FEATURES
from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_temporal_validation as temporal
from src.evaluation import nhanes_threshold_analysis as threshold_stage


@pytest.fixture
def cohorts():
    rng = np.random.default_rng(42)
    def frame(n, offset, cycles):
        df = pd.DataFrame({c: rng.normal(50, 5, n) for c in CORE_FEATURES})
        df["age"] = rng.integers(20, 80, n)
        for c, values in [("sex", [1., 2.]), ("race_ethnicity", [1., 3., 6.]), ("smoking_status", [0., 1., 2.])]:
            df[c] = rng.choice(values, n)
        df["seqn"] = np.arange(offset, offset+n)
        df["nhanes_cycle"] = cycles
        df["diabetes"] = np.tile([0, 0, 1], n // 3)
        return df
    dev = frame(120, 0, np.repeat(DEVELOPMENT_CYCLES, 30))
    val = frame(60, 1000, "2021_2023")
    dev.loc[::5, "bmi"] = np.nan
    val.loc[0, "bmi"] = np.nan
    val.loc[1, ["bmi", "waist_cm"]] = np.nan
    val.loc[2, ["bmi", "waist_cm", "hdl_mg_dl"]] = np.nan
    val.index = np.arange(1000, 1060)  # Distinguish rows even when checking fits by index.
    return dev, val


@pytest.fixture
def references():
    points, cycles = [], []
    for model in temporal.MODELS:
        for point, target, threshold in [
            ("default_0.50", np.nan, 0.5),
            ("nearest_sensitivity_0.80", 0.8, 0.12345678901234567),
            ("nearest_sensitivity_0.90", 0.9, 0.08765432109876543),
            ("youden_j_maximum", np.nan, 0.13264101080417215),
        ]:
            row = dict(model=model, operating_point=point, threshold=threshold, target_sensitivity=target)
            row.update({c: 0.5 for c in temporal.RATE_COMPARISONS})
            row["sensitivity"] = target if np.isfinite(target) else 0.2
            points.append(row)
            for cycle, delta in zip(DEVELOPMENT_CYCLES, [-0.03, -0.01, 0.01, 0.02]):
                cycles.append(dict(model=model, operating_point=point, held_out_cycle=cycle,
                                   sensitivity=row["sensitivity"]+delta, specificity=0.5+delta))
    threshold_frame = pd.DataFrame(points)
    frames = dict(thresholds=threshold_frame, cycles=pd.DataFrame(cycles),
        nested=pd.DataFrame([dict(model=m, roc_auc_mean=0.81, average_precision_mean=0.42,
                                  brier_score_mean=0.10) for m in temporal.MODELS]),
        calibration=pd.DataFrame([dict(model=m, calibration="raw", roc_auc_mean=0.8,
            average_precision_mean=0.4, brier_score_mean=0.1, log_loss_mean=0.32,
            calibration_intercept_mean=0.02, calibration_slope_mean=1.02, ece_mean=0.01)
            for m in temporal.MODELS]))
    fixed = threshold_frame[["model", "operating_point", "threshold", "target_sensitivity"]]
    return fixed, frames


def test_preflight_separation_columns_missingness(cohorts):
    dev, val = cohorts
    audit = temporal.audit_cohorts(dev, val)
    assert audit["seqn_overlap_n"] == 0
    assert audit["cohorts"]["development"]["n"] == 120
    assert audit["cohorts"]["temporal_validation"]["prevalence"] == pytest.approx(1/3)
    assert audit["validation_missing_field_counts"] == {"0": 57, "1": 1, "2": 1, "3+": 1}
    assert audit["cohorts"]["temporal_validation"]["feature_missingness"]["bmi"]["n"] == 3
    overlap = val.copy()
    overlap.iloc[0, overlap.columns.get_loc("seqn")] = dev.seqn.iloc[0]
    with pytest.raises(ValueError, match="overlap"):
        temporal.audit_cohorts(dev, overlap)
    with pytest.raises(ValueError, match="missing required columns"):
        temporal.audit_cohorts(dev, val.drop(columns="bmi"))
    with pytest.raises(ValueError, match="missing required columns"):
        temporal.audit_cohorts(dev, val.drop(columns="diabetes"))


def test_frozen_fits_development_only_metrics_and_schemas(cohorts, references, tmp_path, monkeypatch):
    dev, val = cohorts
    fixed, frames = references
    original_dev, original_val = dev.copy(deep=True), val.copy(deep=True)
    actual_factory = temporal.make_provisional_pipeline
    fit_rows, preprocessing_rows, prediction_rows = [], [], []
    prep_fit = ColumnTransformer.fit_transform
    def record_preprocessing(self, X, y=None, **kwargs):
        assert set(X.index) == set(dev.index) and set(X.index).isdisjoint(val.index)
        preprocessing_rows.append(set(X.index))
        return prep_fit(self, X, y, **kwargs)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", record_preprocessing)
    def tracked(model):
        pipeline = actual_factory(model)
        assert isinstance(pipeline, Pipeline)
        params = pipeline.named_steps["model"].get_params()
        assert params.get("class_weight") is None and params["random_state"] == 42
        if model == "LogisticRegression":
            assert params["C"] == 0.1 and params["solver"] == "lbfgs" and params["max_iter"] == 2000
            assert params["l1_ratio"] == 0
        else:
            for k, v in dict(n_estimators=200, learning_rate=0.03, max_depth=4, min_child_weight=10,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=10,
                    objective="binary:logistic", scale_pos_weight=1).items():
                assert params[k] == v
        fit, predict = pipeline.fit, pipeline.predict_proba
        def record_fit(X, y):
            assert set(X.index) == set(dev.index) and set(X.index).isdisjoint(val.index)
            fit_rows.append((model, tuple(X.index)))
            result = fit(X, y)
            numeric = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
            np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, dev[NUMERIC_FEATURES].median())
            return result
        def record_predict(X):
            assert set(X.index) == set(val.index)
            probability = predict(X)
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            prediction_rows.append((model, probability[:, 1]))
            return probability
        pipeline.fit, pipeline.predict_proba = record_fit, record_predict
        return pipeline
    monkeypatch.setattr(temporal, "make_provisional_pipeline", tracked)
    def forbidden(*args, **kwargs):
        pytest.fail("No calibration, tuning, or threshold selection allowed")
    monkeypatch.setattr(CalibratedClassifierCV, "fit", forbidden)
    monkeypatch.setattr(GridSearchCV, "fit", forbidden)
    monkeypatch.setattr(RandomizedSearchCV, "fit", forbidden)
    monkeypatch.setattr(threshold_stage, "candidate_operating_points", forbidden)
    reader = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Unexpected cohort read during evaluation"))
    metrics, points, curve, missingness = temporal.evaluate_temporal(dev, val, fixed, frames, tmp_path)
    assert len(fit_rows) == len(preprocessing_rows) == len(prediction_rows) == 2
    assert len(metrics) == 2 and len(points) == 8 and len(curve) == 20 and len(missingness) == 4
    for model, probability in prediction_rows:
        row = metrics[metrics.model == model].iloc[0]
        assert row.roc_auc == pytest.approx(roc_auc_score(val.diabetes, probability))
        assert row.average_precision == pytest.approx(average_precision_score(val.diabetes, probability))
        assert row.brier_score == pytest.approx(brier_score_loss(val.diabetes, probability.astype(float)))
        assert row.difference_vs_nested_roc_auc == pytest.approx(row.roc_auc-0.81)
        assert row.fit_count == 1
        for _, point in points[points.model == model].iterrows():
            prior = fixed[(fixed.model == model) & (fixed.operating_point == point.operating_point)].iloc[0]
            assert point.threshold == prior.threshold
            predicted = probability.astype(float) >= point.threshold
            assert point.tp == ((val.diabetes.to_numpy() == 1) & predicted).sum()
            assert point.fp == ((val.diabetes.to_numpy() == 0) & predicted).sum()
            assert point.tp + point.fp + point.tn + point.fn == len(val)
            if pd.notna(point.target_sensitivity):
                assert point.absolute_sensitivity_target_deviation == pytest.approx(abs(point.sensitivity-point.target_sensitivity))
    assert set(["roc_auc", "average_precision", "brier_score", "log_loss", "calibration_intercept",
                "calibration_slope", "ece", "mean_predicted_probability", "observed_prevalence"]) <= set(metrics)
    assert {"difference_vs_development_oof_sensitivity", "cycle_sensitivity_min", "cycle_sensitivity_max",
            "distance_outside_cycle_sensitivity_range", "precision", "negative_predictive_value"} <= set(points)
    for group, expected_n in [("all_core10_observed", 57), ("one_or_more_core10_missing", 3)]:
        assert missingness[missingness.completeness_group == group].n.eq(expected_n).all()
    for name, frame in [("metrics", metrics), ("thresholds", points), ("calibration_curve", curve),
                        ("missingness_robustness", missingness)]:
        saved = reader(tmp_path / f"temporal_validation_{name}.csv")
        assert set(saved) == set(frame)
    pd.testing.assert_frame_equal(dev, original_dev)
    pd.testing.assert_frame_equal(val, original_val)


def test_completeness_empty_and_single_class():
    frame = pd.DataFrame({c: [1., 2.] for c in CORE_FEATURES})
    frame["diabetes"] = [0, 0]
    output = temporal.completeness_metrics(frame, [0.1, 0.2], "LR")
    assert output.iloc[0].outcome_status == "one_class" and pd.isna(output.iloc[0].roc_auc)
    assert pd.isna(output.iloc[0].average_precision)
    assert output.iloc[0].brier_score == pytest.approx(0.025)
    assert output.iloc[1].n == 0 and output.iloc[1].outcome_status == "empty"


def test_fixed_cohort_paths_and_counts(cohorts, monkeypatch):
    dev, val = cohorts
    frozen_dev = pd.concat([dev.iloc[[0]]] * 25580, ignore_index=True)
    frozen_val = pd.concat([val.iloc[[0]]] * 7536, ignore_index=True)
    for frame, offset, positives in [(frozen_dev, 0, 3690), (frozen_val, 100000, 1070)]:
        frame["seqn"] = np.arange(offset, offset+len(frame))
        frame["diabetes"] = 0
        frame.loc[:positives-1, "diabetes"] = 1
    calls = []
    def read(path):
        calls.append(path)
        if path == temporal.DEVELOPMENT_PATH:
            return frozen_dev
        assert path == temporal.VALIDATION_PATH
        return frozen_val
    monkeypatch.setattr(pd, "read_csv", read)
    _, _, audit = temporal.load_cohorts()
    assert calls == [temporal.DEVELOPMENT_PATH, temporal.VALIDATION_PATH]
    assert audit["cohorts"]["temporal_validation"]["n"] == 7536
    frozen_val.loc[0, "diabetes"] = 0
    with pytest.raises(ValueError, match="frozen cohort counts"):
        temporal.load_cohorts()


def test_preflight_recording_frozen_inputs_and_prior_results(cohorts, references, tmp_path, monkeypatch):
    dev, val = cohorts
    fixed, frames = references
    dev_path, val_path = tmp_path / "dev.csv", tmp_path / "val.csv"
    dev_path.write_text("frozen development")
    val_path.write_text("frozen temporal")
    paths = {}
    for name, frame in frames.items():
        path = tmp_path / ("prior_"+name+".csv")
        frame.to_csv(path, index=False)
        paths[name] = path
    all_paths = [dev_path, val_path, *paths.values()]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_paths}
    monkeypatch.setattr(temporal, "DEVELOPMENT_PATH", dev_path)
    monkeypatch.setattr(temporal, "VALIDATION_PATH", val_path)
    monkeypatch.setattr(temporal, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(temporal, "REFERENCE_PATHS", paths)
    monkeypatch.setattr(temporal, "load_cohorts", lambda: (dev, val, temporal.audit_cohorts(dev, val)))
    # Exercise real reference loading, including full-precision thresholds.
    loaded_fixed, _ = temporal.load_references()
    pd.testing.assert_frame_equal(loaded_fixed, fixed)
    def evaluate(d, v, loaded, refs):
        preflight = json.loads((tmp_path / "temporal_validation_provenance.json").read_text())
        assert preflight["status"] == "preflight_complete" and preflight["preflight"]["seqn_overlap_n"] == 0
        assert d is dev and v is val
        pd.testing.assert_frame_equal(loaded, fixed)
        return pd.DataFrame([dict(model=m, fit_warnings="[]") for m in temporal.MODELS]), None, None, None
    monkeypatch.setattr(temporal, "evaluate_temporal", evaluate)
    temporal.run_temporal_validation()
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_paths}
    provenance = json.loads((tmp_path / "temporal_validation_provenance.json").read_text())
    assert provenance["status"] == "complete"
    assert provenance["sha256_before"] == provenance["sha256_after"]
    assert provenance["prior_result_sha256_before"] == provenance["prior_result_sha256_after"]
    assert provenance["fixed_thresholds"][1]["threshold"] == fixed.iloc[1].threshold

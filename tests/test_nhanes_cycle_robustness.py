import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data.nhanes_model_preprocessing import NUMERIC_FEATURES
from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_cycle_robustness as robustness
from src.evaluation import nhanes_preprocessing_comparison as development
from src.evaluation import nhanes_threshold_analysis as threshold_analysis


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({c: rng.normal(50, 5, 160) for c in CORE_FEATURES})
    df["age"] = rng.integers(20, 80, len(df))
    for col, values in [("sex", [1., 2.]), ("race_ethnicity", [1., 3., 6.]),
                        ("smoking_status", [0., 1., 2.])]:
        df[col] = rng.choice(values, len(df))
    df.loc[::4, ["bmi", "sex"]] = np.nan
    df["seqn"] = np.arange(len(df))
    df["nhanes_cycle"] = np.repeat(robustness.DEVELOPMENT_CYCLES, 40)
    df["diabetes"] = np.tile([0, 0, 0, 1], 40)
    return df


@pytest.fixture
def threshold_file(tmp_path):
    rows = []
    for model in robustness.MODELS:
        for point, target, threshold in [
            ("default_0.50", np.nan, 0.5),
            ("nearest_sensitivity_0.80", 0.8, 0.12345678901234567),
            ("nearest_sensitivity_0.90", 0.9, 0.08765432109876543),
            ("youden_j_maximum", np.nan, 0.1543210987654321),
        ]:
            rows.append(dict(model=model, operating_point=point, threshold=threshold, target_sensitivity=target))
    path = tmp_path / "prior_thresholds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_threshold_loading_full_precision_and_validation(threshold_file, tmp_path):
    loaded = robustness.load_fixed_thresholds(threshold_file)
    expected = pd.read_csv(threshold_file, float_precision="round_trip")
    pd.testing.assert_frame_equal(loaded, expected)
    assert loaded.loc[loaded.operating_point == "nearest_sensitivity_0.80", "threshold"].iloc[0] == 0.12345678901234567
    for problem in ["duplicate", "missing", "invalid", "wrong_default", "missing_column"]:
        bad = expected.copy()
        if problem == "duplicate":
            bad = pd.concat([bad, bad.iloc[:1]])
        elif problem == "missing":
            bad = bad.iloc[1:]
        elif problem == "invalid":
            bad.loc[1, "threshold"] = 1.5
        elif problem == "wrong_default":
            bad.loc[0, "threshold"] = 0.51
        else:
            bad = bad.drop(columns="threshold")
        bad_path = tmp_path / (problem + ".csv")
        bad.to_csv(bad_path, index=False)
        with pytest.raises(ValueError):
            robustness.load_fixed_thresholds(bad_path)


def test_real_cycle_fits_fixed_parameters_and_thresholds(sample, threshold_file, tmp_path, monkeypatch):
    fixed = robustness.load_fixed_thresholds(threshold_file)
    original = sample.copy(deep=True)
    threshold_hash = hashlib.sha256(threshold_file.read_bytes()).hexdigest()
    factory = robustness.make_provisional_pipeline
    calls, preprocessing_rows = [], []
    prep_fit = ColumnTransformer.fit_transform
    def tracked_preprocessing(self, X, y=None, **kwargs):
        preprocessing_rows.append(set(X.index))
        return prep_fit(self, X, y, **kwargs)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", tracked_preprocessing)
    def tracked(model):
        pipeline = factory(model)
        assert isinstance(pipeline, Pipeline)
        fit, predict = pipeline.fit, pipeline.predict_proba
        expected_test = set()
        def record_fit(X, y):
            cycle = robustness.DEVELOPMENT_CYCLES[len(calls) // 2 % 4]
            train = sample[sample.nhanes_cycle != cycle]
            test = sample[sample.nhanes_cycle == cycle]
            expected_test.update(test.index)
            assert set(X.index) == set(train.index)
            assert len(set(train.nhanes_cycle)) == 3
            assert train.seqn.isin(test.seqn).sum() == 0
            params = pipeline.named_steps["model"].get_params()
            assert params.get("class_weight") is None
            if model == "LogisticRegression":
                assert params["C"] == 0.1 and params["max_iter"] == 2000 and params["solver"] == "lbfgs"
            else:
                for key, value in dict(n_estimators=200, learning_rate=0.03, max_depth=4,
                        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=1.0, reg_lambda=10.0, scale_pos_weight=1.0, random_state=42).items():
                    assert params[key] == value
            calls.append((model, cycle, set(X.index)))
            result = fit(X, y)
            assert preprocessing_rows[-1] == set(train.index)
            statistics = pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"].statistics_
            np.testing.assert_allclose(statistics, train[NUMERIC_FEATURES].median())
            return result
        def record_predict(X):
            assert set(X.index) == expected_test
            prediction = predict(X)
            assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
            return prediction
        pipeline.fit, pipeline.predict_proba = record_fit, record_predict
        return pipeline
    monkeypatch.setattr(robustness, "make_provisional_pipeline", tracked)
    def fail(*args, **kwargs):
        pytest.fail("Threshold reselection or calibration must not run")
    monkeypatch.setattr(threshold_analysis, "candidate_operating_points", fail)
    monkeypatch.setattr(CalibratedClassifierCV, "fit", fail)
    real_read = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Unexpected cohort access"))
    metrics, thresholds, summary = robustness.compare_cycles(sample, fixed, tmp_path)
    assert len(calls) == 8 and len(preprocessing_rows) == 8
    for i in range(0, 8, 2):
        assert calls[i][1:] == calls[i+1][1:]
    assert len(metrics) == 8 and len(thresholds) == 32
    assert set(robustness.FREE_METRICS) | {"test_n", "positive_n", "negative_n", "prevalence"} <= set(metrics)
    assert {"tp", "fp", "tn", "fn", "threshold", "sensitivity", "specificity", "precision",
            "negative_predictive_value", "balanced_accuracy", "predicted_positive_fraction",
            "absolute_sensitivity_target_deviation"} <= set(thresholds)
    assert metrics.fit_warnings.eq("[]").all()
    assert metrics.test_n.eq(40).all() and metrics.train_n.eq(120).all()
    assert metrics.positive_n.eq(10).all() and metrics.negative_n.eq(30).all()
    for _, row in thresholds.iterrows():
        prior = fixed[(fixed.model == row.model) & (fixed.operating_point == row.operating_point)].iloc[0]
        assert row.threshold == prior.threshold
        assert row.tp + row.fp + row.tn + row.fn == 40
        if pd.notna(row.target_sensitivity):
            assert row.absolute_sensitivity_target_deviation == pytest.approx(abs(row.sensitivity-row.target_sensitivity))
    for _, row in summary.iterrows():
        if row.scope == "threshold_free":
            values = metrics.loc[metrics.model == row.model, row.metric]
        else:
            values = thresholds.loc[(thresholds.model == row.model) &
                                    (thresholds.operating_point == row.operating_point), row.metric]
        assert row.cycles_n == 4
        assert row.valid_cycles_n == values.notna().sum()
        if values.isna().any():
            # Undefined PPV in any cycle must not be silently omitted from means.
            assert pd.isna(row["mean"]) and pd.isna(row["std"]) and pd.isna(row["range"])
        else:
            assert row["mean"] == pytest.approx(values.mean())
            assert row["std"] == pytest.approx(values.std(ddof=1))
            assert row["range"] == pytest.approx(values.max()-values.min())
    for name, frame in [("metrics", metrics), ("thresholds", thresholds), ("summary", summary)]:
        saved = real_read(tmp_path / f"cycle_robustness_{name}.csv")
        assert set(saved) == set(frame)
    pd.testing.assert_frame_equal(sample, original)
    assert threshold_hash == hashlib.sha256(threshold_file.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="all four"):
        robustness.compare_cycles(sample[sample.nhanes_cycle != "2011_2012"], fixed, tmp_path)
    sample["nhanes_cycle"] = "2021_2023"
    with pytest.raises(ValueError, match="separate"):
        robustness.compare_cycles(sample, fixed, tmp_path)


def test_development_loader_only(monkeypatch, sample):
    frame = pd.concat([sample.iloc[[0]]] * 25580, ignore_index=True)
    frame["seqn"] = np.arange(len(frame))
    frame["diabetes"] = 0
    frame.loc[:3689, "diabetes"] = 1
    calls = []
    def read(path):
        calls.append(path)
        assert path == development.DEVELOPMENT_PATH
        return frame
    monkeypatch.setattr(pd, "read_csv", read)
    robustness.load_development_data()
    assert calls == [development.DEVELOPMENT_PATH]


def test_provenance_frozen_hashes_and_fixed_threshold_source(sample, threshold_file, tmp_path, monkeypatch):
    # Synthetic files; do not read actual temporal-validation data even to hash it.
    dev, temporal = tmp_path / "development.csv", tmp_path / "temporal_sentinel.csv"
    dev.write_text("frozen development")
    temporal.write_text("frozen temporal")
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal, threshold_file)]
    monkeypatch.setattr(robustness, "DEVELOPMENT_PATH", dev)
    monkeypatch.setattr(robustness, "THRESHOLD_PATH", threshold_file)
    monkeypatch.setattr(robustness, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(robustness, "load_development_data", lambda: sample)
    expected = robustness.load_fixed_thresholds(threshold_file)
    def compare(df, fixed):
        assert df is sample
        pd.testing.assert_frame_equal(fixed, expected)
        return pd.DataFrame(columns=["model", "held_out_cycle", "fit_warnings"]), None, None
    monkeypatch.setattr(robustness, "compare_cycles", compare)
    robustness.run_cycle_robustness()
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal, threshold_file)]
    provenance = json.loads((tmp_path / "cycle_robustness_provenance.json").read_text())
    assert provenance["sha256_before"] == provenance["sha256_after"]
    assert provenance["threshold_sha256_before"] == provenance["threshold_sha256_after"]
    assert provenance["threshold_source"] == str(threshold_file)
    assert provenance["fixed_operating_points"][1]["threshold"] == expected.iloc[1].threshold

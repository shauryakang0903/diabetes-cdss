import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_threshold_analysis as thresholds
from src.evaluation import nhanes_preprocessing_comparison as development


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({c: rng.normal(50, 5, 90) for c in CORE_FEATURES})
    df["age"] = rng.integers(20, 80, len(df))
    for col, values in [("sex", [1., 2.]), ("race_ethnicity", [1., 3., 6.]),
                        ("smoking_status", [0., 1., 2.])]:
        df[col] = rng.choice(values, len(df))
    df.loc[::4, ["bmi", "sex"]] = np.nan
    df["seqn"] = np.arange(len(df))
    df["nhanes_cycle"] = "2011_2012"
    df["diabetes"] = np.tile([0, 0, 1], 30)
    return df


def test_confusion_metrics_and_boundaries():
    y, p = [0, 0, 1, 1], [0.1, 0.6, 0.8, 0.2]
    frame = thresholds.threshold_metrics(y, p, [0, 0.5, 1])
    row = frame.iloc[1]
    assert row[["tp", "fp", "tn", "fn"]].tolist() == [1, 1, 1, 1]
    for metric in ["sensitivity", "specificity", "precision", "negative_predictive_value",
                   "accuracy", "f1", "balanced_accuracy", "false_positive_rate",
                   "false_negative_rate", "predicted_positive_fraction"]:
        assert row[metric] == pytest.approx(0.5)
    assert row.youden_j == 0
    assert pd.isna(frame.iloc[0].negative_predictive_value)
    assert pd.isna(frame.iloc[2].precision)
    assert frame.iloc[2].f1 == 0
    # Equality is positive, including exact ties and probabilities at 0 or 1.
    tied = thresholds.threshold_metrics([0, 1, 0, 1], [0, 0.5, 0.5, 1], [0, 0.5, 1])
    assert tied.iloc[1][["tp", "fp", "tn", "fn"]].tolist() == [2, 1, 1, 0]
    assert tied.iloc[2][["tp", "fp", "tn", "fn"]].tolist() == [1, 0, 2, 1]


def test_vectorized_counts_agree_with_sklearn():
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, 200)
    p = rng.choice([0, 0.1, 0.3, 0.5, 0.9, 1], 200)
    for _, row in thresholds.threshold_metrics(y, p, [0, 0.1, 0.15, 0.3, 0.5, 0.9, 1]).iterrows():
        tn, fp, fn, tp = confusion_matrix(y, p >= row.threshold, labels=[0, 1]).ravel()
        assert row[["tp", "fp", "tn", "fn"]].tolist() == [tp, fp, tn, fn]


def test_deterministic_exact_candidate_selection():
    y, p = [0, 0, 1, 1], [0.1, 0.6, 0.8, 0.2]
    first = thresholds.candidate_operating_points(y, p).set_index("operating_point")
    second = thresholds.candidate_operating_points(y[::-1], p[::-1]).set_index("operating_point")
    pd.testing.assert_frame_equal(first, second)
    assert set(first.index) == set(thresholds.POINTS)
    assert first.loc["default_0.50", "threshold"] == 0.5
    for point in ["nearest_sensitivity_0.80", "nearest_sensitivity_0.90"]:
        assert first.loc[point, "threshold"] == 0.2
        assert first.loc[point, "sensitivity"] == 1  # Nearest attainable, not exact target.
    assert first.loc["youden_j_maximum", "threshold"] == 0.8  # Higher specificity breaks tie.
    assert first.loc["nearest_sensitivity_specificity", "threshold"] == 0.6
    # Check that candidate selection is not restricted to the display grid.
    exact = thresholds.candidate_operating_points([0, 1, 0, 1, 1, 1, 1],
        [0.123456, 0.654321, 0.111111, 0.7, 0.8, 0.9, 0.01]).set_index("operating_point")
    assert exact.loc["nearest_sensitivity_0.80", "threshold"] == 0.654321
    assert exact.loc["nearest_sensitivity_0.80", "sensitivity"] == 0.8


def test_oof_rows_fit_separation_schemas_reproducibility(sample, tmp_path, monkeypatch):
    original = sample.copy(deep=True)
    factory = thresholds.make_provisional_pipeline
    expected = list(StratifiedKFold(5, shuffle=True, random_state=42).split(sample, sample.diabetes))
    calls = []
    def tracked(model):
        pipeline = factory(model)
        assert isinstance(pipeline, Pipeline) and "preprocessing" in pipeline.named_steps
        if model == "XGBoost":
            pipeline.set_params(model__n_estimators=3)
        fit, predict = pipeline.fit, pipeline.predict_proba
        training_ids = set()
        def record_fit(X, y):
            fold = len(calls) % 5
            train, test = expected[fold]
            assert set(X.index) == set(train)
            training_ids.update(X.index)
            calls.append((model, tuple(X.index)))
            return fit(X, y)
        def record_predict(X):
            assert len(X) == 18 and set(X.index).isdisjoint(training_ids)
            return predict(X)
        pipeline.fit, pipeline.predict_proba = record_fit, record_predict
        return pipeline
    monkeypatch.setattr(thresholds, "make_provisional_pipeline", tracked)
    real_read = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Unexpected cohort read"))
    curve, points, oof, diagnostics = thresholds.analyze_thresholds(sample, tmp_path)
    repeated = thresholds.analyze_thresholds(sample, tmp_path)
    for a, b in zip((curve, points, oof), repeated[:3]):
        pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(sample, original)
    assert len(calls) == 20
    for start in [0, 10]:
        assert [rows for _, rows in calls[start:start+5]] == [rows for _, rows in calls[start+5:start+10]]
    assert len(oof) == 2 * len(sample) and not oof.duplicated(["model", "seqn"]).any()
    assert oof.probability.between(0, 1).all() and np.isfinite(oof.probability).all()
    for _, group in oof.groupby("model"):
        assert set(group.seqn) == set(sample.seqn)
        assert group.groupby("seqn").size().eq(1).all()
        assert group.groupby("outer_fold").size().eq(18).all()
    assert len(curve) == 1962 and len(points) == 10
    assert len(diagnostics) == 10 and all(d["warnings"] == [] for d in diagnostics)
    required = {"model", "threshold", "tp", "fp", "tn", "fn", "sensitivity", "specificity",
                "precision", "negative_predictive_value", "accuracy", "f1", "balanced_accuracy",
                "false_positive_rate", "false_negative_rate", "predicted_positive_fraction"}
    assert required <= set(curve) and required | {"operating_point", "target_sensitivity"} <= set(points)
    assert (points[["tp", "fp", "tn", "fn"]].sum(axis=1) == len(sample)).all()
    for name, frame in [("threshold_curve", curve), ("threshold_candidate_points", points),
                        ("threshold_oof_predictions", oof)]:
        saved = real_read(tmp_path / (name + ".csv"))
        assert set(saved) == set(frame)
    sample["nhanes_cycle"] = "2021_2023"
    with pytest.raises(ValueError, match="separate"):
        thresholds.analyze_thresholds(sample, tmp_path)


def test_development_only_loader(monkeypatch, sample):
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
    thresholds.load_development_data()
    assert calls == [development.DEVELOPMENT_PATH]


def test_hash_integrity_and_provenance(tmp_path, monkeypatch, sample):
    # Synthetic sentinels, never actual temporal-validation data.
    dev, temporal = tmp_path / "development.csv", tmp_path / "temporal_sentinel.csv"
    dev.write_text("frozen development")
    temporal.write_text("frozen temporal")
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)]
    monkeypatch.setattr(thresholds, "DEVELOPMENT_PATH", dev)
    monkeypatch.setattr(thresholds, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(thresholds, "load_development_data", lambda: sample)
    monkeypatch.setattr(thresholds, "analyze_thresholds", lambda df: (None, None, None, []))
    thresholds.run_threshold_analysis()
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)]
    provenance = json.loads((tmp_path / "threshold_analysis_provenance.json").read_text())
    assert provenance["sha256_before"] == provenance["sha256_after"]
    assert provenance["calibration"] == "none" and provenance["seed"] == 42


def test_invalid_thresholds():
    for grid in [[], [-0.1], [1.1], [np.nan]]:
        with pytest.raises(ValueError, match="Thresholds"):
            thresholds.threshold_metrics([0, 1], [0.1, 0.9], grid)

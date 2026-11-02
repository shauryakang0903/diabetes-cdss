import hashlib
import json
from functools import wraps

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_calibration_comparison as calibration
from src.evaluation import nhanes_preprocessing_comparison as development


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({c: rng.normal(50, 5, 120) for c in CORE_FEATURES})
    df["age"] = rng.integers(20, 80, len(df))
    for col, values in [("sex", [1., 2.]), ("race_ethnicity", [1., 3., 6.]),
                        ("smoking_status", [0., 1., 2.])]:
        df[col] = rng.choice(values, len(df))
    df.loc[::4, ["bmi", "sex"]] = np.nan
    df["seqn"] = np.arange(len(df))
    df["nhanes_cycle"] = "2011_2012"
    df["diabetes"] = rng.binomial(1, expit(-0.7 + (df.age - 50) / 30))
    return df


@pytest.mark.parametrize("model", calibration.MODELS)
@pytest.mark.parametrize("method", calibration.METHODS)
def test_candidate_construction(model, method):
    candidate = calibration.make_candidate(model, method, StratifiedKFold(3))
    if method == "raw":
        pipeline = candidate
        assert isinstance(candidate, Pipeline)
    else:
        assert isinstance(candidate, CalibratedClassifierCV)
        assert candidate.method == method and candidate.ensemble is False
        pipeline = candidate.estimator
    assert isinstance(pipeline.named_steps["preprocessing"], ColumnTransformer)
    params = pipeline.named_steps["model"].get_params()
    assert params.get("class_weight") is None
    if model == "LogisticRegression":
        assert params["C"] == 0.1 and params["solver"] == "lbfgs"
        assert "scaler" in pipeline.named_steps["preprocessing"].transformers[0][1].named_steps
    else:
        assert params["n_estimators"] == 200 and params["max_depth"] == 4
        assert params["learning_rate"] == 0.03 and params["scale_pos_weight"] == 1


def test_calibration_metrics_convention_and_bins():
    # At each p, construct exactly the matching empirical event rate. The
    # unpenalized joint logistic MLE is therefore intercept 0 and slope 1.
    y = np.concatenate([np.r_[np.ones(k), np.zeros(10 - k)] for k in (2, 5, 8)])
    probability = np.repeat([0.2, 0.5, 0.8], 10)
    a, b = calibration.calibration_line(y, probability)
    assert a == pytest.approx(0, abs=1e-6) and b == pytest.approx(1, abs=1e-6)
    scores, points = calibration.evaluate_probabilities(y, probability)
    assert set(calibration.METRICS) == set(scores)
    assert scores["ece"] == pytest.approx(0, abs=1e-15)
    assert len(points) == 10 and points.n.sum() == len(y)
    # A known altered logit scale should recover its inverse diagnostic line.
    shifted = expit((np.log(probability / (1 - probability)) - 0.3) / 0.7)
    a, b = calibration.calibration_line(y, shifted)
    assert a == pytest.approx(0.3, abs=1e-6) and b == pytest.approx(0.7, abs=1e-6)
    edges, ece = calibration.reliability_bins([0, 1, 0, 1], [0, 0.1, 0.9, 1])
    assert edges.n.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert ece == pytest.approx(0.45)
    clipped_scores, _ = calibration.evaluate_probabilities(y, np.repeat([0, 0.5, 1], 10))
    assert all(np.isfinite(clipped_scores[m]) for m in calibration.METRICS)
    with pytest.warns(UserWarning, match="Constant predictions"):
        a, b = calibration.calibration_line([0, 1], [0.5, 0.5])
    assert np.isnan(a) and np.isnan(b)
    with pytest.raises(ValueError, match="finite and in"):
        calibration.evaluate_probabilities([0, 1], [np.nan, 2])


def test_real_fit_separation_reproducibility_files(sample, tmp_path, monkeypatch):
    original = sample.copy(deep=True)
    outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(sample, sample.diabetes))
    factory = calibration.make_provisional_pipeline
    def small(model):
        pipeline = factory(model)
        if model == "XGBoost":
            pipeline.set_params(model__n_estimators=3, model__min_child_weight=1)
        return pipeline
    monkeypatch.setattr(calibration, "make_provisional_pipeline", small)
    fitted, predicted, calls = [], [], []
    fit = Pipeline.fit
    def tracked_fit(self, X, y, **kwargs):
        self.training_rows_for_test = set(X.index)
        fitted.append(set(X.index))
        return fit(self, X, y, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", tracked_fit)
    for method in ("predict_proba", "decision_function"):
        original_method = getattr(Pipeline, method)
        @wraps(original_method)
        def tracked_prediction(self, X, _method=original_method, **kwargs):
            assert set(X.index).isdisjoint(self.training_rows_for_test)
            predicted.append(set(X.index))
            return _method(self, X, **kwargs)
        # Preserve Pipeline's conditional method availability and method name:
        # calibration must still use LR decision scores and XGB probabilities.
        monkeypatch.setattr(Pipeline, method, available_if(
            lambda self, _name=method: hasattr(self._final_estimator, _name)
        )(tracked_prediction))
    candidate_factory = calibration.make_candidate
    def tracked_candidate(model, method, inner):
        estimator = candidate_factory(model, method, inner)
        estimator_fit = estimator.fit
        def record(X, y):
            fold = len(calls) // 6 % 5
            train, test = outer[fold]
            assert set(X.index) == set(train) and set(X.index).isdisjoint(test)
            permitted = [set(train)]
            for a, b in inner:
                assert set(a).isdisjoint(b) and set(a) | set(b) == set(range(len(train)))
                permitted.append(set(X.index[a]))
                assert set(X.index[b]).isdisjoint(test)
            start_fit, start_prediction = len(fitted), len(predicted)
            result = estimator_fit(X, y)
            assert all(rows in permitted for rows in fitted[start_fit:])
            if method != "raw":
                assert len(fitted) - start_fit == 4  # three OOF fits plus outer-train refit
                oof_rows = predicted[start_prediction:]
                assert len(oof_rows) == 3
                assert set.union(*oof_rows) == set(train)
                assert sum(map(len, oof_rows)) == len(train)
            calls.append((model, method, tuple(X.index)))
            return result
        estimator.fit = record
        return estimator
    monkeypatch.setattr(calibration, "make_candidate", tracked_candidate)
    read_csv = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Unexpected dataset read"))
    # This intentionally small fixture can collapse an isotonic fit to a
    # constant. Confirm that it is surfaced rather than assigned a fake slope.
    with pytest.warns(UserWarning, match="Constant predictions"):
        folds, summary, curves = calibration.compare_calibration(sample, tmp_path)
        repeated, repeat_summary, repeat_curves = calibration.compare_calibration(sample, tmp_path)
    pd.testing.assert_frame_equal(folds, repeated)
    pd.testing.assert_frame_equal(summary, repeat_summary)
    pd.testing.assert_frame_equal(curves, repeat_curves)
    pd.testing.assert_frame_equal(sample, original)
    assert len(folds) == 30 and len(summary) == 6 and len(curves) == 360
    assert set(calibration.METRICS) <= set(folds)
    ordinary = [m for m in calibration.METRICS if m not in ("calibration_slope", "calibration_intercept")]
    assert np.isfinite(folds[ordinary]).all().all()
    unidentified = folds.calibration_slope.isna()
    assert folds.loc[unidentified, "warnings"].str.contains("Constant predictions").all()
    assert folds.probability_zero_n.ge(0).all() and folds.probability_one_n.ge(0).all()
    for start in range(0, len(calls), 6):
        assert len({c[2] for c in calls[start:start + 6]}) == 1
    for candidate, group in folds.groupby("candidate"):
        row = summary[summary.candidate == candidate].iloc[0]
        for metric in calibration.METRICS:
            if group[metric].isna().any():
                assert np.isnan(row[metric + "_mean"]) and np.isnan(row[metric + "_std"])
            else:
                assert row[metric + "_mean"] == pytest.approx(group[metric].mean())
                assert row[metric + "_std"] == pytest.approx(group[metric].std(ddof=1))
    for name, frame in [("calibration_comparison_folds", folds),
                        ("calibration_comparison_summary", summary), ("calibration_curve_points", curves)]:
        saved = read_csv(tmp_path / (name + ".csv"))
        assert set(saved) == set(frame)
    assert curves[curves.scope == "pooled_outer_oof"].groupby("candidate").n.sum().eq(len(sample)).all()
    sample["nhanes_cycle"] = "2021_2023"
    with pytest.raises(ValueError, match="separate"):
        calibration.compare_calibration(sample, tmp_path)


def test_development_only_loader(monkeypatch, sample):
    frame = pd.concat([sample.iloc[[0]]] * 25580, ignore_index=True)
    frame["seqn"] = np.arange(len(frame))
    frame["diabetes"] = 0
    frame.loc[:3689, "diabetes"] = 1
    reads = []
    def read(path):
        reads.append(path)
        assert path == development.DEVELOPMENT_PATH
        return frame
    monkeypatch.setattr(pd, "read_csv", read)
    calibration.load_development_data()
    assert reads == [development.DEVELOPMENT_PATH]


def test_frozen_hashes_and_provenance(tmp_path, monkeypatch, sample):
    # Synthetic sentinels only; never access the actual temporal-validation CSV.
    dev, temporal = tmp_path / "development.csv", tmp_path / "temporal_sentinel.csv"
    dev.write_text("frozen development")
    temporal.write_text("frozen temporal")
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)]
    monkeypatch.setattr(calibration, "DEVELOPMENT_PATH", dev)
    monkeypatch.setattr(calibration, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(calibration, "load_development_data", lambda: sample)
    monkeypatch.setattr(calibration, "compare_calibration", lambda df: (None, None, None))
    calibration.run_calibration_comparison()
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)]
    provenance = json.loads((tmp_path / "calibration_comparison_provenance.json").read_text())
    assert provenance["sha256_before"] == provenance["sha256_after"]
    assert provenance["ensemble"] is False and provenance["inner_folds"] == 3

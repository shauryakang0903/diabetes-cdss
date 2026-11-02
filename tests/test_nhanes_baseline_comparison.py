import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_baseline_comparison as baseline
from src.evaluation import nhanes_preprocessing_comparison as development


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({c: rng.normal(50, 5, 60) for c in CORE_FEATURES})
    df["age"] = rng.integers(20, 80, 60)
    for col, values in [("sex", [1., 2.]), ("race_ethnicity", [1., 3., 6.]),
                        ("smoking_status", [0., 1., 2.])]:
        df[col] = rng.choice(values, 60)
    df.loc[::4, ["bmi", "sex", "smoking_status"]] = np.nan
    df["seqn"] = np.arange(60)
    df["nhanes_cycle"] = "2011_2012"
    df["diabetes"] = np.tile([0, 0, 1], 20)
    return df


@pytest.mark.parametrize("name,model_type", [("LogisticRegression", LogisticRegression),
                                            ("XGBoost", XGBClassifier)])
def test_pipeline_missing_probabilities(name, model_type, sample):
    pipeline = baseline.make_baseline_pipeline(name)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["model"], model_type)
    prep = pipeline.named_steps["preprocessing"]
    assert isinstance(prep, ColumnTransformer)
    assert all(isinstance(t[1], Pipeline) for t in prep.transformers)
    numeric = prep.transformers[0][1]
    assert numeric.named_steps["imputer"].strategy == "median"
    assert ("scaler" in numeric.named_steps) == (name == "LogisticRegression")
    assert prep.transformers[1][1].named_steps["encoder"].handle_unknown == "ignore"
    pipeline.fit(sample[CORE_FEATURES], sample.diabetes)
    assert np.isfinite(prep.transform(sample[CORE_FEATURES])).all()
    probs = pipeline.predict_proba(sample[CORE_FEATURES])
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    params = pipeline.named_steps["model"].get_params()
    assert params.get("class_weight") is None
    if name == "XGBoost":
        assert params["scale_pos_weight"] == 1 and params["n_jobs"] == 1


def test_folds_schema_and_immutable_input(sample, tmp_path, monkeypatch):
    original = sample.copy(deep=True)
    calls = []
    factory = baseline.make_baseline_pipeline
    def tracked(name):
        pipeline = factory(name)
        fit = pipeline.fit
        def record(X, y):
            calls.append((name, tuple(X.index)))
            assert len(X) == 48
            return fit(X, y)
        pipeline.fit = record
        return pipeline
    monkeypatch.setattr(baseline, "make_baseline_pipeline", tracked)
    # No dataset reads are allowed during comparison of an already-loaded frame.
    real_read = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Unexpected data read"))
    folds, summary = baseline.compare_baselines(sample, tmp_path)
    assert [ids for _, ids in calls[:5]] == [ids for _, ids in calls[5:]]
    assert len(calls) == 10 and len(folds) == 10 and len(summary) == 2
    assert set(folds.model) == set(baseline.MODELS)
    required = {"model", "fold", "train_n", "validation_n", "train_positive_rate",
                "validation_positive_rate", "model_parameters", "fit_warnings"}
    assert required | set(baseline.BASELINE_METRICS) <= set(folds)
    for model in baseline.MODELS:
        group = folds[folds.model == model]
        assert group.fold.tolist() == [1, 2, 3, 4, 5]
        assert group.validation_positive_rate.eq(1 / 3).all()
        for metric in baseline.BASELINE_METRICS:
            row = summary[summary.model == model].iloc[0]
            assert row[metric + "_mean"] == pytest.approx(group[metric].mean())
            assert row[metric + "_std"] == pytest.approx(group[metric].std(ddof=1))
    for suffix, expected in [("folds", folds), ("summary", summary)]:
        saved = real_read(tmp_path / f"baseline_model_comparison_{suffix}.csv")
        assert set(saved) == set(expected)
        assert json.loads(saved.feature_list.iloc[0]) == CORE_FEATURES
        assert json.loads(saved.model_parameters.iloc[0])["class_weight"] is None
    pd.testing.assert_frame_equal(sample, original)
    sample["nhanes_cycle"] = "2021_2023"
    with pytest.raises(ValueError, match="separate"):
        baseline.compare_baselines(sample, tmp_path)


def test_fixed_loader_reads_only_development(monkeypatch, sample):
    frozen = pd.concat([sample.iloc[[0]]] * 25580, ignore_index=True)
    frozen["seqn"] = np.arange(len(frozen))
    frozen["diabetes"] = 0
    frozen.loc[:3689, "diabetes"] = 1
    reads = []
    def read(path):
        reads.append(path)
        assert path.name == "nhanes_development_2011_2020.csv"
        return frozen
    monkeypatch.setattr(pd, "read_csv", read)
    baseline.load_development_data()
    assert reads == [development.DEVELOPMENT_PATH]


def test_runner_preserves_frozen_files(tmp_path, monkeypatch, sample):
    # Synthetic sentinels: no actual temporal-validation data is read by this test.
    dev = tmp_path / "development.csv"
    temporal = tmp_path / "temporal_sentinel.csv"
    dev.write_text("frozen development sentinel")
    temporal.write_text("frozen temporal sentinel")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)}
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setattr(baseline, "DEVELOPMENT_PATH", dev)
    monkeypatch.setattr(baseline, "OUTPUT_DIR", output)
    monkeypatch.setattr(baseline, "load_development_data", lambda: sample)
    def compare(df):
        assert df is sample
        return "folds", "summary"
    monkeypatch.setattr(baseline, "compare_baselines", compare)
    assert baseline.run_baseline_comparison() == ("folds", "summary")
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)}
    provenance = json.loads((output / "baseline_model_comparison_provenance.json").read_text())
    assert provenance["sha256_before"] == provenance["sha256_after"]

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.nhanes_model_preprocessing import (
    STRATEGIES, NUMERIC_FEATURES, make_preprocessor,
)
from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_preprocessing_comparison as experiment


@pytest.fixture
def sample():
    rng = np.random.default_rng(12)
    n = 60
    df = pd.DataFrame({col: rng.normal(50, 5, n) for col in NUMERIC_FEATURES})
    df["age"] = rng.integers(20, 80, n)
    df["sex"] = rng.choice([1., 2.], n)
    df["race_ethnicity"] = rng.choice([1., 3., 6.], n)
    df["smoking_status"] = rng.choice([0., 1., 2.], n)
    df.loc[::5, ["bmi", "hdl_mg_dl", "sex", "smoking_status"]] = np.nan
    df["seqn"] = np.arange(n)
    df["nhanes_cycle"] = "2011_2012"
    df["diabetes"] = np.tile([0, 0, 1], n // 3)
    return df


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_missing_and_unknown(strategy, sample):
    pipeline = experiment.make_probe_pipeline(strategy)
    assert isinstance(pipeline, Pipeline)
    transformer = pipeline.named_steps["preprocessing"]
    assert isinstance(transformer, ColumnTransformer)
    assert all(isinstance(item[1], Pipeline) for item in transformer.transformers)
    pipeline.fit(sample[CORE_FEATURES], sample.diabetes)
    transformed = transformer.transform(sample[CORE_FEATURES])
    assert np.isfinite(transformed).all()
    unseen = sample.iloc[:2][CORE_FEATURES].copy()
    unseen[["sex", "race_ethnicity", "smoking_status"]] = 999
    assert np.isfinite(transformer.transform(unseen)).all()
    assert np.isfinite(pipeline.predict_proba(unseen)).all()
    assert isinstance(transformer.named_transformers_["numeric"].named_steps["scaler"], StandardScaler)
    if strategy == "B_iterative":
        assert isinstance(transformer.named_transformers_["numeric"].named_steps["imputer"], IterativeImputer)
    assert "scaler" not in make_preprocessor(strategy, branch="xgboost").transformers[0][1].named_steps


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_missing_numeric_column(strategy, sample):
    sample["hdl_mg_dl"] = np.nan
    transformer = make_preprocessor(strategy)
    result = transformer.fit_transform(sample[CORE_FEATURES])
    assert np.isfinite(result).all()
    assert result.shape[1] == len(transformer.get_feature_names_out())


def test_training_statistics_only(sample):
    train = sample.iloc[:40].copy()
    held_out = sample.iloc[40:].copy()
    transformer = make_preprocessor()
    transformer.fit(train[CORE_FEATURES])
    imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    statistics = imputer.statistics_.copy()
    np.testing.assert_allclose(statistics, train[NUMERIC_FEATURES].median())
    held_out[NUMERIC_FEATURES] = 1e8
    transformer.transform(held_out[CORE_FEATURES])
    np.testing.assert_array_equal(statistics, imputer.statistics_)


def test_loader_development_only(monkeypatch, sample):
    frozen = pd.concat([sample.iloc[[0]]] * 25580, ignore_index=True)
    frozen["seqn"] = np.arange(len(frozen))
    frozen["diabetes"] = 0
    frozen.loc[:3689, "diabetes"] = 1
    calls = []
    def read(path):
        calls.append(path)
        return frozen
    monkeypatch.setattr(pd, "read_csv", read)
    experiment.load_development_data()
    assert calls == [experiment.DEVELOPMENT_PATH]
    assert experiment.DEVELOPMENT_PATH.name == "nhanes_development_2011_2020.csv"
    frozen["nhanes_cycle"] = "2021_2023"
    with pytest.raises(ValueError, match="separate"):
        experiment.load_development_data()


def test_results_and_fold_fit_boundaries(sample, tmp_path, monkeypatch):
    original = sample.copy(deep=True)
    train_ids = []
    actual = experiment.make_probe_pipeline
    def tracked(strategy, seed):
        pipeline = actual(strategy, seed)
        fit = pipeline.fit
        def record(X, y):
            train_ids.append((strategy, tuple(X.index)))
            assert len(X) == 48
            return fit(X, y)
        pipeline.fit = record
        return pipeline
    monkeypatch.setattr(experiment, "make_probe_pipeline", tracked)
    folds, summary = experiment.compare_preprocessing(sample, tmp_path)
    assert len(train_ids) == 10
    assert [ids for _, ids in train_ids[:5]] == [ids for _, ids in train_ids[5:]]
    assert len(folds) == 10 and len(summary) == 2
    assert set(folds.strategy) == set(STRATEGIES)
    assert set(experiment.METRICS) <= set(folds)
    assert folds[list(experiment.METRICS)].apply(lambda s: s.between(0, 1).all()).all()
    for metric in experiment.METRICS:
        for strategy in STRATEGIES:
            values = folds.loc[folds.strategy == strategy, metric]
            row = summary.loc[summary.strategy == strategy].iloc[0]
            assert row[metric + "_mean"] == pytest.approx(values.mean())
            assert row[metric + "_std"] == pytest.approx(values.std(ddof=1))
    for name in ["folds", "summary"]:
        saved = pd.read_csv(tmp_path / f"preprocessing_comparison_{name}.csv")
        assert set(saved.strategy) == set(STRATEGIES)
        assert saved.random_seed.eq(42).all() and saved.fold_count.eq(5).all()
        assert json.loads(saved.feature_list.iloc[0]) == CORE_FEATURES
    pd.testing.assert_frame_equal(sample, original)
    again, _ = experiment.compare_preprocessing(sample, tmp_path)
    pd.testing.assert_frame_equal(folds, again)
    sample["nhanes_cycle"] = "2021_2023"
    with pytest.raises(ValueError, match="separate"):
        experiment.compare_preprocessing(sample, tmp_path)


def test_specificity_and_brier():
    result = experiment.score_predictions([0, 0, 1, 1], [0.1, 0.6, 0.8, 0.2])
    assert result["specificity"] == 0.5
    assert result["recall"] == 0.5
    assert result["brier_score"] == pytest.approx((0.01 + 0.36 + 0.04 + 0.64) / 4)


def test_single_outcome_rejected(sample, tmp_path):
    sample["diabetes"] = 0
    with pytest.raises(ValueError, match="both binary outcome classes"):
        experiment.compare_preprocessing(sample, tmp_path)
    assert not list(tmp_path.iterdir())

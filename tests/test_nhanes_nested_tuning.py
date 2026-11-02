import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.preprocessing import CORE_FEATURES
from src.evaluation import nhanes_nested_tuning as nested
from src.evaluation import nhanes_preprocessing_comparison as development


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({c: rng.normal(50, 3, 60) for c in CORE_FEATURES})
    df["age"] = rng.integers(20, 80, 60)
    for col, values in [("sex", [1., 2.]), ("race_ethnicity", [1., 3., 6.]),
                        ("smoking_status", [0., 1., 2.])]:
        df[col] = rng.choice(values, 60)
    df.loc[::4, ["bmi", "sex"]] = np.nan
    df["seqn"] = np.arange(60)
    df["nhanes_cycle"] = "2011_2012"
    df["diabetes"] = np.tile([0, 0, 1], 20)
    return df


def test_search_construction_and_valid_spaces():
    inner = StratifiedKFold(3, shuffle=True, random_state=42)
    for name in nested.MODELS:
        search = nested.make_search(name, inner)
        assert search.scoring == "roc_auc" and search.refit and search.error_score == "raise"
        pipeline = search.estimator
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps["preprocessing"], ColumnTransformer)
        params = pipeline.get_params()
        space = nested.LR_SPACE if name == "LogisticRegression" else nested.XGB_SPACE
        assert set(space) <= set(params)
        assert pipeline.named_steps["model"].get_params().get("class_weight") is None
        if name == "XGBoost":
            assert search.n_iter == 20 and search.random_state == 42
            assert params["model__scale_pos_weight"] == 1


def test_real_nested_search_separation_reproducibility_schema(sample, tmp_path, monkeypatch):
    # Tiny actual searches for tests; production constants remain 5 LR and 20 XGB candidates.
    monkeypatch.setattr(nested, "LR_SPACE", {"model__C": [0.1, 1.0]})
    monkeypatch.setattr(nested, "XGB_SPACE", {**nested.XGB_SPACE,
        "model__n_estimators": [2], "model__max_depth": [2]})
    monkeypatch.setattr(nested, "XGB_ITERATIONS", 2)
    expected = list(StratifiedKFold(5, shuffle=True, random_state=42).split(sample, sample.diabetes))
    original = sample.copy(deep=True)
    factory = nested.make_search
    seen, fitted_ids = [], []
    pipeline_fit = Pipeline.fit
    def record_pipeline(self, X, y, **kwargs):
        fitted_ids.append(set(X.index))
        assert "preprocessing" in self.named_steps
        return pipeline_fit(self, X, y, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", record_pipeline)
    def tracked(name, inner):
        search = factory(name, inner)
        fit = search.fit
        def record(X, y):
            fold = len(seen) // 2 % 5
            train, test = expected[fold]
            assert set(X.index) == set(train)
            assert set(X.index).isdisjoint(test)
            # Indices passed to search CV are relative to outer-training rows only.
            permitted = [set(train)]
            for a, b in inner:
                assert set(a).isdisjoint(b)
                assert set(a) | set(b) == set(range(len(train)))
                permitted.append(set(X.index[a]))
                assert set(X.index[a]).isdisjoint(test)
                assert set(X.index[b]).isdisjoint(test)
            before = len(fitted_ids)
            result = fit(X, y)
            assert all(ids in permitted for ids in fitted_ids[before:])
            for key, value in search.best_params_.items():
                space = nested.LR_SPACE if name == "LogisticRegression" else nested.XGB_SPACE
                assert value in space[key]
            seen.append((name, tuple(X.index)))
            return result
        search.fit = record
        return search
    monkeypatch.setattr(nested, "make_search", tracked)
    real_read = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Unexpected dataset access"))
    folds, summary, stability = nested.nested_tuning(sample, tmp_path)
    repeated, _, _ = nested.nested_tuning(sample, tmp_path)
    pd.testing.assert_frame_equal(folds, repeated)
    pd.testing.assert_frame_equal(sample, original)
    assert len(folds) == 10 and len(summary) == 2
    for i in range(0, len(seen), 2):
        assert seen[i][1] == seen[i + 1][1]
    required = {"model", "outer_fold", "train_n", "test_n", "train_positive_rate",
                "test_positive_rate", "best_inner_roc_auc", "selected_hyperparameters"}
    assert required | set(nested.BASELINE_METRICS) <= set(folds)
    assert folds.fit_warnings.eq("[]").all()
    for name, group in folds.groupby("model"):
        row = summary[summary.model == name].iloc[0]
        for metric in nested.BASELINE_METRICS:
            assert group[metric].between(0, 1).all()
            assert row[metric + "_mean"] == pytest.approx(group[metric].mean())
            assert row[metric + "_std"] == pytest.approx(group[metric].std(ddof=1))
    assert stability.loc[stability.record_type == "frequency", "selection_count"].sum() == 5
    for suffix, frame in [("outer_folds", folds), ("summary", summary), ("hyperparameters", stability)]:
        saved = real_read(tmp_path / f"nested_tuning_{suffix}.csv")
        assert set(saved) == set(frame)
    assert not (tmp_path / "baseline_model_comparison_folds.csv").exists()
    sample["nhanes_cycle"] = "2021_2023"
    with pytest.raises(ValueError, match="separate"):
        nested.nested_tuning(sample, tmp_path)


def test_fixed_development_loader(monkeypatch, sample):
    frame = pd.concat([sample.iloc[[0]]] * 25580, ignore_index=True)
    frame["seqn"] = np.arange(len(frame))
    frame["diabetes"] = 0
    frame.loc[:3689, "diabetes"] = 1
    def read(path):
        assert path == development.DEVELOPMENT_PATH
        assert path.name == "nhanes_development_2011_2020.csv"
        return frame
    monkeypatch.setattr(pd, "read_csv", read)
    assert len(nested.load_development_data()) == 25580


def test_frozen_file_integrity_and_provenance(tmp_path, monkeypatch, sample):
    # Synthetic sentinels only, never the actual temporal file.
    dev, temporal = tmp_path / "dev.csv", tmp_path / "temporal_sentinel.csv"
    dev.write_text("frozen development")
    temporal.write_text("frozen temporal")
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)]
    monkeypatch.setattr(nested, "DEVELOPMENT_PATH", dev)
    monkeypatch.setattr(nested, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(nested, "load_development_data", lambda: sample)
    monkeypatch.setattr(nested, "nested_tuning", lambda df: (None, None, None))
    nested.run_nested_tuning()
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (dev, temporal)]
    provenance = json.loads((tmp_path / "nested_tuning_provenance.json").read_text())
    assert provenance["sha256_before"] == provenance["sha256_after"]

from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import pytest

from src.data import load_nhanes as loader
from src.data.preprocessing import (
    ACTIVITY_COLUMNS, CORE_FEATURES, clean_diabetes_outcome,
    clean_activity_variables, derive_smoking_status, preprocess_nhanes,
)


@pytest.fixture(scope="module")
def raw_cycles():
    # Real local XPT files: missing research inputs must fail, never silently skip.
    return {cycle: loader.load_cycle(cycle) for cycle in
            (*loader.DEVELOPMENT_CYCLES, loader.VALIDATION_CYCLE)}


@pytest.fixture(scope="module")
def cohorts(raw_cycles):
    dev = preprocess_nhanes(pd.concat(
        [raw_cycles[c] for c in loader.DEVELOPMENT_CYCLES], ignore_index=True))
    val = preprocess_nhanes(raw_cycles[loader.VALIDATION_CYCLE])
    return dev, val


@pytest.mark.parametrize("cycle", (*loader.DEVELOPMENT_CYCLES, loader.VALIDATION_CYCLE))
def test_load_cycle(cycle, raw_cycles):
    df = raw_cycles[cycle]
    assert not df.empty
    assert df.seqn.is_unique and df.seqn.notna().all()
    assert set(df.nhanes_cycle) == {cycle}
    assert set(CORE_FEATURES[:-1] + ["seqn", "diabetes_status",
           "smoking_lifetime", "smoking_current"]) <= set(df)
    assert set(ACTIVITY_COLUMNS).isdisjoint(df) if cycle == loader.VALIDATION_CYCLE else set(ACTIVITY_COLUMNS) <= set(df)


@pytest.mark.parametrize("index,n,positive", [(0, 25580, 3690), (1, 7536, 1070)])
def test_cohort_counts_and_cleaning(index, n, positive, cohorts):
    df = cohorts[index]
    assert len(df) == n
    assert df.diabetes.sum() == positive
    assert df.age.ge(20).all()
    assert set(df.diabetes) == {0, 1}
    assert df.seqn.is_unique
    assert set(CORE_FEATURES) <= set(df)
    assert df.smoking_status.dropna().isin([0, 1, 2]).all()
    assert df[CORE_FEATURES].isna().any().any()  # No imputation.
    if index == 0:
        assert df[ACTIVITY_COLUMNS].stack().dropna().isin([0, 1]).all()
    else:
        assert set(ACTIVITY_COLUMNS).isdisjoint(df)


def test_separate_cohorts(cohorts, raw_cycles):
    dev, val = cohorts
    loader.validate_cohort_separation(dev, val)
    assert set(dev.seqn).isdisjoint(val.seqn)
    with pytest.raises(ValueError, match="separate"):
        preprocess_nhanes(pd.concat([raw_cycles[c] for c in raw_cycles]))
    with pytest.raises(ValueError, match="separate"):
        loader.validate_cohort(val, "development")
    overlapping = val.copy()
    overlapping.loc[overlapping.index[0], "seqn"] = dev.seqn.iloc[0]
    with pytest.raises(ValueError, match="overlap"):
        loader.validate_cohort_separation(dev, overlapping)


def test_development_loader_only_development(monkeypatch, raw_cycles):
    calls = []
    def load(cycle):
        calls.append(cycle)
        return raw_cycles[cycle]
    monkeypatch.setattr(loader, "load_cycle", load)
    loader.load_all_development_cycles()
    assert calls == list(loader.DEVELOPMENT_CYCLES)
    assert loader.VALIDATION_CYCLE not in loader.CYCLE_CONFIG


def test_outcome_mapping():
    result = clean_diabetes_outcome(pd.DataFrame({"diabetes_status": [1, 2, 3, 7, 9, np.nan]}))
    assert result.diabetes.iloc[:2].tolist() == [1, 0]
    assert result.diabetes.iloc[2:].isna().all()


def test_smoking_mapping():
    df = pd.DataFrame({"smoking_lifetime": [2, 1, 1, 1, 7, 1, np.nan],
                       "smoking_current": [np.nan, 3, 1, 2, 9, 7, 1]})
    result = derive_smoking_status(df)
    assert result.smoking_status.iloc[:4].tolist() == [0, 1, 2, 2]
    assert result.smoking_status.iloc[4:].isna().all()
    assert "smoking_status" not in df


def test_activity_mapping():
    result = clean_activity_variables(pd.DataFrame({c: [1, 2, 7, 9, np.nan] for c in ACTIVITY_COLUMNS}))
    for col in ACTIVITY_COLUMNS:
        assert result[col].iloc[:2].tolist() == [1, 0]
        assert result[col].iloc[2:].isna().all()


@pytest.mark.parametrize("bp_type,prefix", [("legacy", "BPX"), ("oscillometric", "BPXO")])
def test_blood_pressure(bp_type, prefix):
    df = pd.DataFrame({"SEQN": [1, 2],
        **{f"{prefix}SY{i}": [v, np.nan] for i, v in enumerate([120, np.nan, 140], 1)},
        **{f"{prefix}DI{i}": [v, np.nan] for i, v in enumerate([0, 60, np.nan], 1)}})
    result = loader.harmonize_blood_pressure(df, bp_type)
    assert result.mean_systolic_bp.iloc[0] == 130
    assert result.mean_diastolic_bp.iloc[0] == 30
    assert result.iloc[1, 1:].isna().all()


def test_tiny_zero(tmp_path, monkeypatch):
    path = tmp_path / "sample.xpt"
    path.touch()
    monkeypatch.setattr(pd, "read_sas", lambda *a, **k: pd.DataFrame({
        "numeric": [5.397605346934028e-79, 0, 1e-60, np.nan, 2], "text": ["x"] * 5}))
    result = loader.read_xpt(path)
    assert result.numeric.iloc[:3].tolist() == [0, 0, 1e-60]
    assert pd.isna(result.numeric.iloc[3])
    assert result.text.tolist() == ["x"] * 5
    assert path.read_bytes() == b""
    with pytest.raises(FileNotFoundError, match="Missing NHANES file"):
        loader.read_xpt(tmp_path / "missing.xpt")


def test_defensive_preprocessing(raw_cycles):
    raw = raw_cycles["2011_2012"].copy()
    with pytest.raises(ValueError, match="missing required columns.*age"):
        preprocess_nhanes(raw.drop(columns="age"))
    with pytest.raises(ValueError, match="duplicate seqn"):
        preprocess_nhanes(pd.concat([raw, raw.iloc[:1]]))
    raw["age"] = 19
    with pytest.raises(ValueError, match="Empty cohort"):
        preprocess_nhanes(raw)
    with pytest.raises(ValueError, match="empty"):
        preprocess_nhanes(raw.iloc[:0])


def test_adult_boundary(raw_cycles):
    raw = raw_cycles["2011_2012"].iloc[:4].copy()
    raw["age"] = [19, 20, np.nan, 80]
    raw["diabetes_status"] = [1, 2, 1, 3]
    result = preprocess_nhanes(raw)
    assert result.seqn.tolist() == [raw.seqn.iloc[1]]
    assert result.diabetes.tolist() == [0]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "empty"])
def test_component_validation(monkeypatch, fault):
    frame = pd.DataFrame({"SEQN": [1, 1] if fault == "duplicate" else [1]})
    if fault == "empty":
        frame = frame.iloc[:0]
    monkeypatch.setattr(loader, "read_xpt", lambda path: frame)
    with pytest.raises(ValueError, match="missing required columns|duplicate SEQN|empty"):
        loader.load_cycle("2011_2012")


def test_unexpected_codes_surfaced():
    with pytest.warns(UserWarning, match="unexpected categorical codes"):
        result = clean_diabetes_outcome(pd.DataFrame({"diabetes_status": [42]}))
    assert result.diabetes.isna().all()


def test_scripts_import_without_loading(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Import must not read data")
    monkeypatch.setattr(pd, "read_csv", fail)
    monkeypatch.setattr(pd, "read_sas", fail)
    for name in ["build_nhanes_development_cohort", "build_nhanes_validation_2021_2023", "final_audit_nhanes_development"]:
        importlib.import_module("scripts." + name)


def test_export_regression(cohorts):
    for df, name in zip(cohorts, ["nhanes_development_2011_2020", "nhanes_validation_2021_2023"]):
        previous = pd.read_csv(Path("data/processed/nhanes") / (name + ".csv"))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), previous, check_dtype=False, atol=1e-10)

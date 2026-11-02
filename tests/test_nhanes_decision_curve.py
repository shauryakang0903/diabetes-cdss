import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.evaluation import nhanes_decision_curve as dca
from src.models.nhanes_phase2_model import load_metadata, sha256


@pytest.fixture(autouse=True)
def no_fitting_or_artifact_changes(monkeypatch):
    paths = (dca.DEFAULT_ARTIFACT_PATH, dca.DEFAULT_METADATA_PATH, dca.ROOT/"app.py",
             dca.ROOT/"configs/nhanes_recommendation_rules.json")
    before = {p: sha256(p) for p in paths}
    def forbidden(*args, **kwargs):
        pytest.fail("DCA must not fit or refit any model/preprocessing")
    for cls in (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, XGBClassifier, LogisticRegression):
        monkeypatch.setattr(cls, "fit", forbidden)
        if hasattr(cls, "fit_transform"):
            monkeypatch.setattr(cls, "fit_transform", forbidden)
    yield
    assert before == {p: sha256(p) for p in paths}


def test_hand_calculated_model_net_benefit_and_ties():
    curve = dca.net_benefit_curve([1, 1, 0, 0], [.8, .1, .7, .2], [.2])
    row = curve.iloc[0]
    assert (row.tp, row.fp, row.tn, row.fn) == (1, 2, 0, 1)
    assert row.net_benefit == 1/4-2/4*(.2/(1-.2)) == .125
    assert row.sensitivity == .5 and row.specificity == 0


def test_evaluate_all_none_and_exact_identical_strategies():
    thresholds = [.01, .1, .2, .5, .8]
    y = [0, 0, 1, 0, 1, 0]
    all_curve = dca.net_benefit_curve(y, np.ones(6), thresholds)
    none_curve = dca.net_benefit_curve(y, np.zeros(6), thresholds)
    expected = np.array([2/6-(1-2/6)*t/(1-t) for t in thresholds])
    np.testing.assert_array_equal(all_curve.evaluate_none_net_benefit, 0)
    np.testing.assert_allclose(all_curve.evaluate_all_net_benefit, expected)
    np.testing.assert_array_equal(all_curve.net_benefit, all_curve.evaluate_all_net_benefit)
    np.testing.assert_array_equal(none_curve.net_benefit, 0)
    assert (dca.net_benefit_curve([0, 0], [.8, .1], [.2]).net_benefit < 0).all()


@pytest.mark.parametrize("thresholds", [[0], [1], [-.01], [1.1], [np.nan], [np.inf], [], [[.1]]])
def test_invalid_thresholds(thresholds):
    with pytest.raises(ValueError, match="thresholds"):
        dca.net_benefit_curve([0, 1], [.1, .9], thresholds)


def test_probability_and_empty_input_validation():
    with pytest.raises(ValueError, match="participant"):
        dca.net_benefit_curve([], [], [.1])
    with pytest.raises(ValueError, match="Probabilities"):
        dca.net_benefit_curve([0, 1], [-.01, .9], [.1])


def test_exact_frozen_references_grid_and_determinism():
    metadata = load_metadata()
    grid = dca.threshold_grid(metadata)
    assert grid[0] == .01 and grid[-1] == .5 and len(grid) == 102
    assert len(dca.BASE_GRID) == 99
    for name in dca.REFERENCE_ORDER:
        assert metadata["reference_thresholds"][name] in grid
    first = dca.combined_curve([0, 1, 0, 1], [.1, .8, .3, .5], [.2, .7, .4, .6], grid,
                               dataset="development", model_version=metadata["model_version"])
    pd.testing.assert_frame_equal(first, dca.combined_curve([0, 1, 0, 1], [.1, .8, .3, .5], [.2, .7, .4, .6], grid,
                                                         dataset="development", model_version=metadata["model_version"]))
    assert {"threshold", "xgb_net_benefit", "lr_net_benefit", "evaluate_all_net_benefit", "evaluate_none_net_benefit"} <= set(first)


def test_disjoint_sampled_intervals_and_empty_runs():
    t = [.01, .015, .02, .025, .03]
    runs = dca.sampled_intervals(t, [True, False, True, True, False])
    assert [(r["start_threshold"], r["end_threshold"], r["tested_point_count"]) for r in runs] == [(.01, .01, 1), (.02, .025, 2)]
    assert runs[0]["preceding_tested_threshold"] is None
    assert runs[0]["following_tested_threshold"] == .015
    assert dca.sampled_intervals(t, [False]*5) == []
    assert len(dca.sampled_intervals(t, [True]*5)) == 1
    with pytest.raises(ValueError, match="increasing"):
        dca.sampled_intervals([.2, .1], [True, True])


def test_comparator_alignment_checks():
    development = pd.DataFrame(dict(seqn=[10, 20, 30], diabetes=[0, 1, 0]))
    xgb = pd.DataFrame(dict(seqn=[10, 20, 30], diabetes=[0, 1, 0], nhanes_cycle=["2011_2012"]*3,
                            model="XGBoost", outer_fold=[1, 2, 3], probability=[.1, .8, .2]))
    lr = xgb.copy()
    lr["model"] = "LogisticRegression"
    lr["probability"] = [.2, .7, .3]
    saved = pd.concat([xgb, lr.iloc[::-1]], ignore_index=True)
    np.testing.assert_array_equal(dca.align_saved_oof(development, saved, "LogisticRegression"), [.2, .7, .3])
    assert dca.align_saved_oof(development, xgb, "LogisticRegression") is None
    duplicate = pd.concat([saved, lr.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        dca.align_saved_oof(development, duplicate, "LogisticRegression")
    corrupted = saved.copy()
    corrupted.loc[corrupted.model == "LogisticRegression", "outer_fold"] = 5
    with pytest.raises(ValueError, match="outer_fold"):
        dca.align_saved_oof(development, corrupted, "LogisticRegression")


def test_actual_saved_oof_and_frozen_temporal_only(monkeypatch):
    original = dca.subgroup.predict_probabilities
    calls = []
    def spy(frame, *, pipeline):
        assert len(frame) == 7536 and set(frame.nhanes_cycle) == {"2021_2023"}
        calls.append(len(frame))
        return original(frame, pipeline=pipeline)
    monkeypatch.setattr(dca.subgroup, "predict_probabilities", spy)
    inputs, metadata, comparator = dca.load_dca_inputs()
    assert calls == [7536]
    assert len(inputs["development"]["xgb"]) == len(inputs["development"]["lr"]) == 25580
    assert inputs["temporal"]["lr"] is None
    assert comparator["development_lr_available"] and not comparator["temporal_lr_available"]
    assert metadata["model_version"] == "nhanes_phase2_xgb_v1"


def test_full_output_schema_and_no_fake_temporal_lr(tmp_path, monkeypatch):
    prior = tmp_path/"prior"
    (prior/"subgroup_audit").mkdir(parents=True)
    (prior/"subgroup_audit/subgroup_audit_summary.json").write_text("{}", encoding="utf-8")
    out = tmp_path/"decision_curve"
    out.mkdir()
    (out/"independent_test_report.xml").write_text("preserve test report", encoding="utf-8")
    monkeypatch.setattr(dca, "DCA_DIR", out)
    monkeypatch.setattr(dca, "RESULTS_DIR", prior)
    frame = pd.DataFrame(dict(diabetes=[0, 1, 0, 1]))
    inputs = {"development": dict(frame=frame, xgb=np.array([.1, .8, .3, .5]), lr=np.array([.2, .7, .4, .6])),
              "temporal": dict(frame=frame.copy(), xgb=np.array([.1, .7, .2, .4]), lr=None)}
    monkeypatch.setattr(dca, "load_dca_inputs", lambda: (inputs, load_metadata(), dict(temporal_lr_available=False)))
    development, temporal, refs, summary = dca.run_dca(out)
    assert len(development) == len(temporal) == 102 and len(refs) == 12
    assert temporal.lr_net_benefit.isna().all() and not temporal.lr_available.any()
    assert set(refs[refs.dataset == "temporal"].model) == {"XGBoost"}
    assert {"dataset", "threshold_name", "threshold_value", "model", "TP", "FP", "TN", "FN", "sensitivity", "specificity", "net_benefit", "evaluate_all_net_benefit", "evaluate_none_net_benefit"} <= set(refs)
    for row in refs.itertuples():
        assert row.threshold_value == load_metadata()["reference_thresholds"][row.threshold_name]
    assert summary["sha256_before"] == summary["sha256_after"]
    assert not summary["model_retrained"] and not summary["new_threshold_selected"]
    assert "POST-HOC" in summary["temporal_warning"]
    assert not summary["subgroup_specific_dca"] and not summary["subgroup_specific_thresholds"]
    assert len(list((out/"figures").glob("*.png"))) == len(list((out/"figures").glob("*.pdf"))) == 4
    assert len(summary["output_sha256"]) == 11
    assert "independent_test_report.xml" not in summary["output_sha256"]
    json.loads((out/"dca_summary.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="outputs must stay"):
        dca.run_dca(tmp_path/"elsewhere")

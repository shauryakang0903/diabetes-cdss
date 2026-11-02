"""Fast synthetic tests; never open the temporal cohort or its results."""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import CORE_FEATURES
from src.data.nhanes_model_preprocessing import NUMERIC_FEATURES
from src.models.nhanes_tabm import (
    TabMConfig, TabMPreprocessor, experimental_metadata, fit_tabm, predict_probability,
    seed_everything,
)
from src.evaluation import nhanes_tabm_evaluation as evaluation


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({feature: rng.normal(size=120) for feature in CORE_FEATURES})
    frame['sex'] = rng.choice([1., 2.], 120)
    frame['race_ethnicity'] = rng.choice([1., 3., 4.], 120)
    frame['smoking_status'] = rng.choice([0., 1., 2.], 120)
    frame['seqn'] = np.arange(120)
    frame['nhanes_cycle'] = '2011_2012'
    frame['diabetes'] = np.tile([0, 0, 0, 1], 30)
    frame.loc[::11, ['bmi', 'sex', 'hdl_mg_dl']] = np.nan
    return frame


def test_core_schema_and_train_only_preprocessing(sample):
    train, held = sample.iloc[:90], sample.iloc[90:].copy()
    prep = TabMPreprocessor().fit(train)
    median = prep.numeric.named_steps['imputer'].statistics_.copy()
    means = prep.numeric.named_steps['scaler'].mean_.copy()
    held.loc[:, NUMERIC_FEATURES] = 1e6
    held['sex'] = 99
    num, cat = prep.transform(held)
    assert num.shape == (30, 7) and cat.shape == (30, 3)
    assert (cat[:, 0] == 0).all()  # reserved unknown, not a learned category
    np.testing.assert_array_equal(prep.numeric.named_steps['imputer'].statistics_, median)
    np.testing.assert_array_equal(prep.numeric.named_steps['scaler'].mean_, means)
    np.testing.assert_allclose(median, train[NUMERIC_FEATURES].median())
    with pytest.raises(ValueError, match='Missing required'):
        prep.transform(held.drop(columns='age'))


def test_missing_values_and_unknown_categories(sample):
    prep = TabMPreprocessor().fit(sample)
    missing = sample.iloc[:3].copy()
    missing[CORE_FEATURES] = np.nan
    num, cat = prep.transform(missing)
    assert np.isfinite(num).all() and (cat >= 1).all()
    for column, cardinality in zip(cat.T, prep.cardinalities):
        assert (column < cardinality).all()


def test_seed_and_training_reproducibility(sample):
    config = TabMConfig(n_blocks=1, d_block=8, k=2, batch_size=32)
    outputs = []
    for _ in range(2):
        model, prep, diagnostics, history = fit_tabm(
            sample.iloc[:90], sample.diabetes.iloc[:90], config, max_epochs=2,
            stopping_frame=sample.iloc[90:], stopping_target=sample.diabetes.iloc[90:])
        probability = predict_probability(model, prep, sample.iloc[90:], 'cpu')
        assert ((probability >= 0) & (probability <= 1)).all()
        assert 1 <= diagnostics['best_epoch'] <= diagnostics['epochs_trained'] <= 2
        assert history and diagnostics['best_stopping_roc_auc'] is not None
        outputs.append(probability)
    np.testing.assert_array_equal(*outputs)
    seed_everything(42)
    first = torch.rand(4)
    seed_everything(42)
    assert torch.equal(first, torch.rand(4))


def test_fold_integrity(sample):
    counts = np.zeros(len(sample))
    first = evaluation.outer_splits(sample.diabetes)
    second = evaluation.outer_splits(sample.diabetes)
    for fold, ((train, test), (train2, test2)) in enumerate(zip(first, second), 1):
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(test, test2)
        fit, stop = evaluation.training_partition(train, sample.diabetes.to_numpy(), 42+100*fold+90)
        assert not set(fit) & set(stop) and not set(train) & set(test)
        assert set(fit) | set(stop) == set(train)
        counts[test] += 1
    assert (counts == 1).all()
    with pytest.raises(ValueError, match='overlap'):
        evaluation.split_audit(1, 1, np.array([0]), np.array([0]), np.array([1]), np.array([2]))


def test_nested_fit_separation_and_oof(sample, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('A supplied-frame nested experiment must not open any cohort CSV')
    monkeypatch.setattr(pd, 'read_csv', forbidden)
    actual_fit = evaluation.fit_tabm
    calls = []
    def spy(frame, target, config, **kwargs):
        stop = kwargs.get('stopping_frame')
        calls.append((set(frame.index), set(stop.index) if stop is not None else set()))
        assert stop is None or not set(frame.index) & set(stop.index)
        return actual_fit(frame, target, config, **kwargs)
    monkeypatch.setattr(evaluation, 'fit_tabm', spy)
    config = TabMConfig(n_blocks=1, d_block=8, k=2, batch_size=32)
    folds, oof = evaluation.nested_tabm(sample, search=(config,), max_epochs=1,
                                       patience=1, output_dir=tmp_path)
    assert len(calls) == 25  # 3 inner + epoch selection + all-training refit, per outer fold
    for fold, (_, test) in enumerate(evaluation.outer_splits(sample.diabetes)):
        for fit, stop in calls[fold*5:(fold+1)*5]:
            assert not (fit | stop) & set(sample.iloc[test].seqn)
    assert len(oof) == len(sample) and oof.seqn.is_unique
    assert np.isfinite(oof.tabm_probability).all()
    assert set(evaluation.METRICS) <= set(folds)
    assert set(oof) == {'seqn', 'diabetes', 'outer_fold', 'tabm_probability'}
    for name in ['tabm_fold_metrics.csv', 'tabm_training_history.csv', 'tabm_oof_predictions.csv',
                 'tabm_inner_search.json', 'tabm_split_audit.json']:
        assert (tmp_path/name).is_file()
    assert len(json.loads((tmp_path/'tabm_split_audit.json').read_text())) == 15


def test_loader_development_only(monkeypatch):
    read = pd.read_csv
    paths = []
    before = evaluation.checksum(evaluation.DEVELOPMENT_PATH)
    def guard(path, *args, **kwargs):
        assert path == evaluation.DEVELOPMENT_PATH
        paths.append(path)
        return read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', guard)
    frame = evaluation.load_development_data()
    assert len(frame) == 25580 and len(paths) == 1
    assert evaluation.checksum(evaluation.DEVELOPMENT_PATH) == before


def test_forbidden_cycle_and_duplicate_ids(sample, tmp_path):
    sample.loc[0, 'nhanes_cycle'] = '2021_2023'
    with pytest.raises(ValueError, match='development cycles'):
        evaluation.nested_tabm(sample, output_dir=tmp_path)
    sample['nhanes_cycle'] = '2011_2012'
    sample.loc[1, 'seqn'] = sample.loc[0, 'seqn']
    with pytest.raises(ValueError, match='unique'):
        evaluation.nested_tabm(sample, output_dir=tmp_path)


def test_metrics_and_bootstrap():
    y = np.array([0, 0, 1, 1])
    probability = np.array([.1, .6, .4, .9])
    scores = evaluation.score_probability(y, probability)
    assert scores['accuracy'] == scores['sensitivity'] == scores['specificity'] == .5
    assert scores['roc_auc'] == .75
    assert scores['brier_score'] == pytest.approx(.185)
    reference = evaluation.score_probability(y, probability, evaluation.XGB_THRESHOLD)
    assert reference['sensitivity'] == 1 and reference['specificity'] == .5
    same = evaluation.bootstrap_comparison(y, probability, probability, replicates=25)
    assert (same[['observed_difference', 'ci_low', 'ci_high']] == 0).all().all()
    with pytest.raises(ValueError, match='probabilities'):
        evaluation.score_probability(y, [0, .5, .8, 2])


def test_experimental_metadata_and_frozen_integrity():
    metadata = experimental_metadata(TabMConfig())
    assert 'EXPERIMENTAL DEVELOPMENT-ONLY MODEL' in metadata['status']
    assert 'NOT TEMPORALLY VALIDATED' in metadata['status']
    assert not metadata['primary_cdss_model'] and not metadata['temporal_validation']
    assert metadata['core_features'] == CORE_FEATURES
    assert evaluation.checksum(evaluation.ROOT/'models/nhanes/phase2_xgboost_pipeline.joblib') == (
        'fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528')
    assert evaluation.checksum(evaluation.ROOT/'models/nhanes/phase2_xgboost_metadata.json') == (
        '57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3')


def test_lr_log_loss_recovery_uses_archived_training_only(sample, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from src.evaluation.nhanes_baseline_comparison import make_baseline_pipeline
    historical = tmp_path/'results/nhanes'
    historical.mkdir(parents=True)
    archived, saved = [], []
    splits = evaluation.outer_splits(sample.diabetes)
    for fold, (train, test) in enumerate(splits, 1):
        baseline = make_baseline_pipeline('LogisticRegression').set_params(model__C=.1)
        baseline.fit(sample[CORE_FEATURES].iloc[train], sample.diabetes.iloc[train])
        probability = baseline.predict_proba(sample[CORE_FEATURES].iloc[test])[:, 1]
        saved.append(sample.iloc[test][['seqn', 'diabetes']].assign(
            model='LogisticRegression', outer_fold=fold, probability=probability))
        c = 1. if fold == 3 else .1
        if fold == 3:
            baseline.set_params(model__C=c).fit(sample[CORE_FEATURES].iloc[train], sample.diabetes.iloc[train])
            probability = baseline.predict_proba(sample[CORE_FEATURES].iloc[test])[:, 1]
        scores = evaluation.score_probability(sample.diabetes.iloc[test], probability)
        archived.append(dict(model='LogisticRegression', outer_fold=fold,
                             selected_hyperparameters=json.dumps({'model__C': c}),
                             **{k:v for k,v in scores.items() if k != 'log_loss'}))
    pd.DataFrame(archived).to_csv(historical/'nested_tuning_outer_folds.csv', index=False)
    pd.concat(saved).to_csv(historical/'threshold_oof_predictions.csv', index=False)
    out = tmp_path/'new_results'
    out.mkdir()
    pd.DataFrame(dict(model=['LogisticRegression', 'TabM'], log_loss_mean=[np.nan, .3],
                      log_loss_std=[np.nan, .01])).to_csv(out/'tabm_vs_existing_models.csv', index=False)
    monkeypatch.setattr(evaluation, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluation, 'load_development_data', lambda: sample.copy())
    actual_fit, calls = Pipeline.fit, []
    def spy(self, frame, target, **kwargs):
        assert not set(frame.index) & set(splits[2][1])
        calls.append(set(frame.index))
        return actual_fit(self, frame, target, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('LR metric recovery must never fit XGBoost')
    monkeypatch.setattr(Pipeline, 'fit', spy)
    monkeypatch.setattr(XGBClassifier, 'fit', forbidden)
    comparison, audit = evaluation.complete_lr_comparison(out)
    assert len(calls) == 1 and calls[0] == set(splits[2][0])
    assert not comparison.log_loss_mean.isna().any()
    assert all(row['archived_metrics_reproduced'] for row in audit['folds'])
    assert len(pd.read_csv(out/'lr_nested_recovered_oof_predictions.csv')) == len(sample)
    # A verified cache is reusable without retraining any model.
    monkeypatch.setattr(Pipeline, 'fit', forbidden)
    evaluation.complete_lr_comparison(out)


def test_preflight_snapshot_fails_before_training_on_changed_inputs(tmp_path, monkeypatch):
    model_dir = tmp_path/'models/nhanes'
    model_dir.mkdir(parents=True)
    expected = {}
    for name in evaluation.EXPECTED_PRIMARY_SHA256:
        path = model_dir/name
        path.write_text(name)
        expected[name] = evaluation.checksum(path)
    (tmp_path/'app.py').write_text('protected app')
    development = tmp_path/'development.csv'
    development.write_text('development only')
    monkeypatch.setattr(evaluation, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluation, 'OUTPUT_DIR', tmp_path/'results/nhanes/deep_learning')
    monkeypatch.setattr(evaluation, 'DEVELOPMENT_PATH', development)
    monkeypatch.setattr(evaluation, 'EXPECTED_PRIMARY_SHA256', expected)
    assert len(evaluation.prepare_integrity_snapshot()) == 4
    (tmp_path/'app.py').write_text('changed')
    with pytest.raises(ValueError, match='before training'):
        evaluation.prepare_integrity_snapshot()

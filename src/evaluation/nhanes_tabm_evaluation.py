"""Nested development-only TabM evaluation; never loads the temporal cohort."""

import hashlib
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_preprocessing_comparison import DEVELOPMENT_PATH, load_development_data
from src.models.nhanes_tabm import TabMConfig, fit_tabm, predict_probability

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'results/nhanes/deep_learning'
SEED = 42
XGB_THRESHOLD = 0.14026810228824615
METRICS = ('roc_auc', 'average_precision', 'brier_score', 'log_loss', 'accuracy',
           'sensitivity', 'specificity', 'precision', 'f1')
EXPECTED_PRIMARY_SHA256 = {
    'phase2_xgboost_pipeline.joblib': 'fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528',
    'phase2_xgboost_metadata.json': '57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3',
}
# Fixed before examining any outer-test performance. Default plus two compact
# alternatives; the runner can use a smaller predeclared CPU search after timing.
SEARCH = (TabMConfig(), TabMConfig(n_blocks=2, d_block=128, k=8),
          TabMConfig(n_blocks=3, d_block=128, dropout=.2, k=8))


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_integrity_snapshot():
    """Preflight guard, also usable when the results directory does not exist yet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED_PRIMARY_SHA256.items():
        if checksum(ROOT/'models/nhanes'/name) != expected:
            raise ValueError(f'Frozen primary checksum differs before training: {name}')
    path = OUTPUT_DIR/'protected_hashes_before.json'
    if not path.exists():
        files = [ROOT/'app.py', DEVELOPMENT_PATH]
        files += [ROOT/'models/nhanes'/name for name in EXPECTED_PRIMARY_SHA256]
        files += list((ROOT/'configs').rglob('*'))
        files += list((ROOT/'src/decision_support').glob('*.py'))
        files += list((ROOT/'results/nhanes').rglob('*'))
        hashes = {str(p.resolve()): checksum(p) for p in files if p.is_file()
                  and OUTPUT_DIR.resolve() not in p.resolve().parents}
        path.write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    hashes = json.loads(path.read_text(encoding='utf-8'))
    changed = [p for p, expected in hashes.items() if checksum(p) != expected]
    if changed:
        raise ValueError(f'Protected inputs changed before training: {changed}')
    return hashes


def score_probability(y, probability, threshold=.5):
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if (len(y) != len(probability) or set(np.unique(y)) != {0, 1}
            or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all()):
        raise ValueError('Metrics require aligned binary outcomes and finite probabilities in [0,1]')
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y, probability)),
                average_precision=float(average_precision_score(y, probability)),
                brier_score=float(brier_score_loss(y, probability)),
                log_loss=float(log_loss(y, probability, labels=[0, 1])),
                accuracy=float(accuracy_score(y, predicted)), sensitivity=float(tp/(tp+fn)),
                specificity=float(tn/(tn+fp)), precision=float(precision_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)))


def outer_splits(target, folds=5):
    return list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(np.zeros(len(target)), target))


def training_partition(indices, target, seed):
    """Early-stop observations are exclusively inside the caller's training set."""
    fit, stop = train_test_split(np.asarray(indices), test_size=.15, random_state=seed,
                                 stratify=np.asarray(target)[indices])
    assert not np.intersect1d(fit, stop).size
    assert set(np.r_[fit, stop]) == set(indices)
    return fit, stop


def index_digest(indices):
    return hashlib.sha256(np.sort(indices).astype('<i8').tobytes()).hexdigest()


def split_audit(outer, inner, fit, stop, evaluation, outer_test):
    for left, right in [(fit, stop), (fit, evaluation), (stop, evaluation),
                        (fit, outer_test), (stop, outer_test), (evaluation, outer_test)]:
        if np.intersect1d(left, right).size:
            raise ValueError('Nested training/stopping/evaluation participant overlap')
    return dict(outer_fold=outer, inner_fold=inner,
                **{name: dict(n=len(value), row_index_sha256=index_digest(value))
                   for name, value in [('gradient_train', fit), ('early_stop', stop),
                                       ('inner_evaluation', evaluation), ('outer_test', outer_test)]})


def nested_tabm(frame, *, search=SEARCH, max_epochs=200, patience=20, threads=2,
                 device='cpu', output_dir=OUTPUT_DIR):
    """Inner evaluation never supplies gradients, preprocessing or early stopping.

    Selected configs get a training-only epoch determination, then a fresh refit
    on ALL outer-training rows for that fixed epoch count before one test score.
    """
    if not frame.seqn.is_unique or frame.seqn.isna().any():
        raise ValueError('Participant identifiers must be unique and nonmissing')
    if 'nhanes_cycle' in frame and not set(frame.nhanes_cycle).issubset(
            {'2011_2012', '2013_2014', '2015_2016', '2017_2020'}):
        raise ValueError('Only frozen development cycles are permitted')
    X, y = frame[CORE_FEATURES], frame.diabetes.astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1} or np.bincount(y).min() < 5:
        raise ValueError('Five stratified folds require both binary outcomes with at least five each')
    out = Path(output_dir)
    if out.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError('Cannot write results into frozen data')
    out.mkdir(parents=True, exist_ok=True)
    probabilities = np.full(len(frame), np.nan)
    coverage = np.zeros(len(frame), dtype=int)
    fold_labels = np.zeros(len(frame), dtype=int)
    rows, histories, searches, audits = [], [], [], []
    for fold, (train, test) in enumerate(outer_splits(y), 1):
        print(f'Outer fold {fold}/5', flush=True)
        candidate_scores = []
        for candidate, config in enumerate(search):
            scores = []
            for inner, (inner_train, inner_test) in enumerate(
                    StratifiedKFold(3, shuffle=True, random_state=SEED).split(X.iloc[train], y[train]), 1):
                seed = SEED + 100*fold + inner
                fit, stop = training_partition(train[inner_train], y, seed)
                evaluation = train[inner_test]
                audit = split_audit(fold, inner, fit, stop, evaluation, test)
                if candidate == 0:
                    audits.append(audit)
                print(f'  candidate {candidate+1}/{len(search)}, inner {inner}/3: {asdict(config)}', flush=True)
                model, prep, diagnostics, history = fit_tabm(
                    X.iloc[fit], y[fit], config, seed=seed, device=device, threads=threads,
                    max_epochs=max_epochs, patience=patience,
                    stopping_frame=X.iloc[stop], stopping_target=y[stop])
                score = roc_auc_score(y[evaluation], predict_probability(model, prep, X.iloc[evaluation], device))
                scores.append(float(score))
                searches.append(dict(outer_fold=fold, candidate=candidate, inner_fold=inner,
                                     inner_roc_auc=float(score), **diagnostics))
                histories.extend(dict(outer_fold=fold, candidate=candidate, inner_fold=inner,
                                      stage='inner', **row) for row in history)
                print(f'    inner AUC={score:.6f}, best epoch={diagnostics["best_epoch"]}, '
                      f'seconds={diagnostics["seconds"]:.1f}', flush=True)
                del model, prep
            candidate_scores.append(float(np.mean(scores)))
        best = int(np.argmax(candidate_scores))  # first candidate wins an exact tie
        config = search[best]
        seed = SEED + 100*fold + 90
        fit, stop = training_partition(train, y, seed)
        split_audit(fold, 0, fit, stop, np.array([], dtype=int), test)
        model, prep, selection, history = fit_tabm(
            X.iloc[fit], y[fit], config, seed=seed, device=device, threads=threads,
            max_epochs=max_epochs, patience=patience,
            stopping_frame=X.iloc[stop], stopping_target=y[stop])
        histories.extend(dict(outer_fold=fold, candidate=best, inner_fold=0,
                              stage='outer_epoch_selection', **row) for row in history)
        del model, prep
        model, prep, refit, history = fit_tabm(
            X.iloc[train], y[train], config, seed=seed, device=device, threads=threads,
            max_epochs=selection['best_epoch'], patience=patience)
        probability = predict_probability(model, prep, X.iloc[test], device)
        probabilities[test], fold_labels[test] = probability, fold
        coverage[test] += 1
        histories.extend(dict(outer_fold=fold, candidate=best, inner_fold=0,
                              stage='outer_refit', **row) for row in history)
        rows.append(dict(model='TabM', outer_fold=fold, train_n=len(train), test_n=len(test),
                         train_positive_rate=float(y[train].mean()), test_positive_rate=float(y[test].mean()),
                         best_inner_roc_auc=candidate_scores[best], selected_candidate=best,
                         selected_hyperparameters=json.dumps(asdict(config), sort_keys=True),
                         best_epoch=selection['best_epoch'], epochs_trained=selection['epochs_trained'],
                         stopping_roc_auc=selection['best_stopping_roc_auc'],
                         refit_epochs=refit['epochs_trained'],
                         outer_fit_seconds=selection['seconds']+refit['seconds'],
                         **score_probability(y[test], probability),
                         **{f'xgb_reference_{k}': v for k, v in score_probability(y[test], probability, XGB_THRESHOLD).items()}))
        pd.DataFrame(rows).to_csv(out/'tabm_fold_metrics.csv', index=False)
        pd.DataFrame(searches).to_json(out/'tabm_inner_search.json', orient='records', indent=2)
        pd.DataFrame(histories).to_csv(out/'tabm_training_history.csv', index=False)
        print(f'  Fold {fold} complete. Outer AUC={rows[-1]["roc_auc"]:.6f}', flush=True)
        del model, prep
    if not (coverage == 1).all() or not np.isfinite(probabilities).all():
        raise ValueError('OOF predictions must cover every participant exactly once')
    oof = pd.DataFrame(dict(seqn=frame.seqn.to_numpy(), diabetes=y, outer_fold=fold_labels,
                            tabm_probability=probabilities))
    oof.to_csv(out/'tabm_oof_predictions.csv', index=False)
    (out/'tabm_split_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    return pd.DataFrame(rows), oof


def bootstrap_comparison(y, tabm, reference, replicates=1000):
    """IID participant bootstrap of fixed aligned OOF scores, not retraining CV."""
    rng = np.random.default_rng(SEED)
    functions = {'roc_auc': roc_auc_score, 'average_precision': average_precision_score,
                 'brier_score': brier_score_loss}
    differences = {name: [] for name in functions}
    y, tabm, reference = map(np.asarray, (y, tabm, reference))
    if not (len(y) == len(tabm) == len(reference)):
        raise ValueError('Paired bootstrap needs aligned participant predictions')
    for _ in range(replicates):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        for name, function in functions.items():
            differences[name].append(float(function(y[indices], tabm[indices])-
                                           function(y[indices], reference[indices])))
    return pd.DataFrame([dict(metric=name, observed_difference=float(function(y, tabm)-function(y, reference)),
                               bootstrap_mean_difference=float(np.mean(differences[name])),
                               ci_low=float(np.quantile(differences[name], .025)),
                               ci_high=float(np.quantile(differences[name], .975)),
                               replicates=len(differences[name]), seed=SEED,
                               comparison='TabM minus XGBoost; fixed OOF participant bootstrap')
                         for name, function in functions.items()])


def compare_existing(folds, oof, output_dir):
    """Use existing nested metrics; validate OOF alignment before paired inference."""
    out = Path(output_dir)
    existing = pd.read_csv(ROOT/'results/nhanes/nested_tuning_outer_folds.csv')
    saved = pd.read_csv(ROOT/'results/nhanes/threshold_oof_predictions.csv')
    summary = []
    for name, group in list(existing.groupby('model', sort=False)) + [('TabM', folds)]:
        row = dict(model=name, source='nested CV', threshold=.5)
        for metric in METRICS:
            row[metric+'_mean'] = float(group[metric].mean()) if metric in group else None
            row[metric+'_std'] = float(group[metric].std(ddof=1)) if metric in group else None
        summary.append(row)
    xgb = saved[saved.model == 'XGBoost']
    aligned = oof.merge(xgb[['seqn', 'diabetes', 'outer_fold', 'probability']],
                        on='seqn', how='left', suffixes=('', '_reference'), validate='one_to_one')
    if (len(aligned) != len(xgb) or aligned.probability.isna().any()
            or not (aligned.diabetes == aligned.diabetes_reference).all()
            or not (aligned.outer_fold == aligned.outer_fold_reference).all()):
        raise ValueError('Saved XGBoost OOF predictions are not participant/fold aligned')
    # These fixed-reference predictions are usable for the nested comparison only
    # when their full fold metrics reproduce the archived nested XGBoost results.
    losses = []
    for fold, group in aligned.groupby('outer_fold'):
        scores = score_probability(group.diabetes, group.probability)
        archived = existing[(existing.model == 'XGBoost') & (existing.outer_fold == fold)].iloc[0]
        for metric in METRICS:
            if metric in archived and not np.isclose(scores[metric], archived[metric], atol=1e-7, rtol=0):
                raise ValueError(f'Saved XGBoost OOF does not reproduce nested {metric}, fold {fold}')
        losses.append(scores['log_loss'])
    for row in summary:
        if row['model'] == 'XGBoost':
            row['log_loss_mean'], row['log_loss_std'] = float(np.mean(losses)), float(np.std(losses, ddof=1))
    comparison = pd.DataFrame(summary)
    comparison.to_csv(out/'tabm_vs_existing_models.csv', index=False)
    bootstrap = bootstrap_comparison(aligned.diabetes, aligned.tabm_probability, aligned.probability)
    bootstrap.to_csv(out/'tabm_bootstrap_comparison.csv', index=False)
    return comparison, bootstrap


def figures(folds, comparison, history_path, output_dir):
    output_dir = Path(output_dir)/'figures'
    output_dir.mkdir(exist_ok=True)
    for metric in ('roc_auc', 'average_precision', 'brier_score'):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(comparison.model, comparison[metric+'_mean'], yerr=comparison[metric+'_std'], capsize=5)
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title('Development nested CV: mean ± fold SD')
        fig.tight_layout()
        fig.savefig(output_dir/f'model_{metric}_comparison.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(folds.outer_fold, folds.roc_auc, marker='o')
    ax.set(xlabel='Outer fold', ylabel='ROC-AUC', title='TabM held-out development performance')
    fig.tight_layout()
    fig.savefig(output_dir/'tabm_fold_performance.png', dpi=160)
    plt.close(fig)
    history = pd.read_csv(history_path)
    representative = history[(history.outer_fold == 1) & (history.stage == 'outer_epoch_selection')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(representative.epoch, representative.training_loss)
    axes[0].set(xlabel='Epoch', ylabel='Training BCE')
    axes[1].plot(representative.epoch, representative.stopping_roc_auc)
    axes[1].set(xlabel='Epoch', ylabel='Training-derived stopping ROC-AUC')
    fig.tight_layout()
    fig.savefig(output_dir/'representative_training.png', dpi=160)
    plt.close(fig)


def run_experiment(*, search=SEARCH, max_epochs=200, patience=20, threads=2):
    start = time.perf_counter()
    hashes = prepare_integrity_snapshot()
    sources = [Path(__file__), ROOT/'src/models/nhanes_tabm.py', ROOT/'scripts/run_nhanes_tabm_experiment.py',
               ROOT/'requirements-nhanes-tabm.txt']
    (OUTPUT_DIR/'tabm_source_hashes.json').write_text(
        json.dumps({str(p): checksum(p) for p in sources}, indent=2), encoding='utf-8')
    frame = load_development_data()  # sole cohort read; no alternate input path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folds, oof = nested_tabm(frame, search=search, max_epochs=max_epochs, patience=patience,
                             threads=threads, device=device)
    comparison, bootstrap = compare_existing(folds, oof, OUTPUT_DIR)
    figures(folds, comparison, OUTPUT_DIR/'tabm_training_history.csv', OUTPUT_DIR)
    changed = [path for path, expected in hashes.items() if checksum(path) != expected]
    if changed:
        raise RuntimeError(f'Protected files changed: {changed}')
    result = dict(status='EXPERIMENTAL DEVELOPMENT-ONLY; NOT TEMPORALLY VALIDATED; NOT PRIMARY CDSS MODEL',
                  core_features=CORE_FEATURES, target='Physician-diagnosed diabetes from DIQ010',
                  development_path=str(DEVELOPMENT_PATH), development_sha256=checksum(DEVELOPMENT_PATH),
                  participants=len(frame), positives=int(frame.diabetes.sum()),
                  seed=SEED, outer_folds=5, inner_folds=3, inner_seed=SEED,
                  stopping_fraction=.15, stopping_seed='42 + 100*outer_fold + inner_fold; outer selection uses +90',
                  search=[asdict(c) for c in search], max_epochs=max_epochs, patience=patience,
                  class_weight=None, loss='Mean independent ensemble-member binary cross entropy',
                  prediction='Mean of member sigmoid probabilities', device=device, threads=threads,
                  cuda_available=torch.cuda.is_available(), torch_cuda_build=torch.version.cuda,
                  environment=dict(python=platform.python_version(), **{name:version(name) for name in
                                   ['torch','tabm','numpy','pandas','scikit-learn','rtdl_num_embeddings']}),
                  deterministic_algorithms=True, outer_participant_folds_aligned=True,
                  xgboost_oof_reproduces_nested_metrics=True,
                  lr_log_loss_unavailable='Nested LR OOF not saved; fixed C=0.1 differs from nested C=1 in fold 3. Not substituted.',
                  metrics={m:dict(mean=float(folds[m].mean()), sd=float(folds[m].std(ddof=1))) for m in METRICS},
                  selected_hyperparameters=folds[['outer_fold','selected_hyperparameters','best_epoch']].to_dict('records'),
                  xgb_reference_threshold=XGB_THRESHOLD,
                  threshold_note='Developed for XGBoost; descriptive cross-model reference, not optimized for TabM',
                  bootstrap_note='1000 IID participant replicates; fixed OOF scores, no retraining; does not capture all training/CV uncertainty',
                  model_artifact_saved=False, temporal_cohort_accessed=False,
                  protected_hashes_before=hashes,
                  protected_hashes_after={path:checksum(path) for path in hashes},
                  seconds=time.perf_counter()-start)
    (OUTPUT_DIR/'tabm_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)
    return result


def complete_lr_comparison(output_dir=OUTPUT_DIR):
    """Recover missing nested LR log loss by replaying only mismatched saved folds.

    This is reporting, not a new search: archived selected parameters and the
    original training rows are fixed. Never construct or refit an XGBoost model.
    Existing fold metrics must reproduce before probabilities are accepted.
    """
    from src.evaluation.nhanes_baseline_comparison import make_baseline_pipeline

    out = Path(output_dir)
    before = checksum(DEVELOPMENT_PATH)
    frame = load_development_data()
    archived = pd.read_csv(ROOT/'results/nhanes/nested_tuning_outer_folds.csv')
    saved = pd.read_csv(ROOT/'results/nhanes/threshold_oof_predictions.csv')
    losses, recovered, audit = [], [], []
    for fold, (train, test) in enumerate(outer_splits(frame.diabetes), 1):
        original = archived[(archived.model == 'LogisticRegression') & (archived.outer_fold == fold)].iloc[0]
        reference = saved[(saved.model == 'LogisticRegression') & (saved.outer_fold == fold)]
        expected = frame.iloc[test][['seqn', 'diabetes']].copy()
        def align(predictions):
            merged = expected.merge(predictions[['seqn', 'diabetes', 'probability']], on='seqn',
                                     validate='one_to_one', suffixes=('', '_reference'), how='left')
            if (len(merged) != len(predictions) or merged.probability.isna().any()
                    or not (merged.diabetes == merged.diabetes_reference).all()):
                raise ValueError(f'LR reference participants/outcomes differ, fold {fold}')
            return merged.probability.to_numpy()
        def matches(probability):
            scores = score_probability(expected.diabetes, probability)
            return all(np.isclose(scores[m], original[m], atol=1e-7, rtol=0)
                       for m in METRICS if m in original)
        probability = align(reference)
        source = 'original saved OOF'
        if not matches(probability):
            cache = out/f'lr_nested_fold{fold}_recovered_oof.csv'
            if cache.exists():
                probability = align(pd.read_csv(cache))
                source = 'verified replay cache'
            else:
                pipeline = make_baseline_pipeline('LogisticRegression').set_params(
                    **json.loads(original.selected_hyperparameters))
                pipeline.fit(frame[CORE_FEATURES].iloc[train], frame.diabetes.iloc[train].astype(int))
                probability = pipeline.predict_proba(frame[CORE_FEATURES].iloc[test])[:, 1]
                replay = expected.assign(outer_fold=fold, probability=probability)
                replay.to_csv(cache, index=False)
                source = 'replay of archived selected LR fit; no tuning'
            if not matches(probability):
                raise ValueError(f'Replayed LR fails archived metric reproduction, fold {fold}')
        losses.append(float(log_loss(expected.diabetes, probability, labels=[0, 1])))
        recovered.append(expected.assign(outer_fold=fold, probability=probability))
        audit.append(dict(outer_fold=fold, source=source,
                          selected_hyperparameters=json.loads(original.selected_hyperparameters),
                          log_loss=losses[-1], archived_metrics_reproduced=True))
    if checksum(DEVELOPMENT_PATH) != before:
        raise ValueError('Development cohort changed during LR reference recovery')
    pd.concat(recovered, ignore_index=True).to_csv(out/'lr_nested_recovered_oof_predictions.csv', index=False)
    comparison = pd.read_csv(out/'tabm_vs_existing_models.csv')
    comparison.loc[comparison.model == 'LogisticRegression', 'log_loss_mean'] = np.mean(losses)
    comparison.loc[comparison.model == 'LogisticRegression', 'log_loss_std'] = np.std(losses, ddof=1)
    comparison.to_csv(out/'tabm_vs_existing_models.csv', index=False)
    report = dict(method='Use archived OOF where matching; replay only mismatched LR folds with archived parameters',
                  development_sha256=before, folds=audit, xgboost_loaded_or_fitted=False,
                  log_loss_mean=float(np.mean(losses)), log_loss_std=float(np.std(losses, ddof=1)))
    (out/'lr_nested_recovery_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    path = out/'tabm_summary.json'
    if path.exists():
        summary = json.loads(path.read_text(encoding='utf-8'))
        summary.pop('lr_log_loss_unavailable', None)
        summary['lr_log_loss_recovery'] = report
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print('Verified nested LR log loss:', report['log_loss_mean'], '+/-', report['log_loss_std'], flush=True)
    return comparison, report


def finalize_tabm_report(output_dir=OUTPUT_DIR):
    """Reproducible reporting only: no TabM fits, search, threshold or calibration changes."""
    out = Path(output_dir)
    comparison, _ = complete_lr_comparison(out)
    path = out/'tabm_summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    folds = pd.read_csv(out/'tabm_fold_metrics.csv')
    oof = pd.read_csv(out/'tabm_oof_predictions.csv')
    inner = pd.read_json(out/'tabm_inner_search.json')
    if (len(oof) != summary['participants'] or not oof.seqn.is_unique
            or not oof.tabm_probability.between(0, 1).all()
            or set(oof.diabetes) != {0, 1} or int(oof.diabetes.sum()) != summary['positives']):
        raise ValueError('Saved TabM OOF integrity check failed')
    baseline = comparison[comparison.model == 'XGBoost'].iloc[0]
    tabm = comparison[comparison.model == 'TabM'].iloc[0]
    summary['mean_fold_differences_vs_xgboost'] = {
        m:float(tabm[m+'_mean']-baseline[m+'_mean']) for m in METRICS}
    summary['paired_bootstrap'] = pd.read_csv(out/'tabm_bootstrap_comparison.csv').to_dict('records')
    summary['comparison_mean_sd'] = comparison.to_dict('records')
    summary['reference_threshold_metrics'] = {
        m:dict(mean=float(folds['xgb_reference_'+m].mean()),
               sd=float(folds['xgb_reference_'+m].std(ddof=1))) for m in METRICS}
    summary['oof_integrity'] = dict(rows=len(oof), unique_participants=int(oof.seqn.nunique()),
        positives=int(oof.diabetes.sum()), probability_min=float(oof.tabm_probability.min()),
        probability_max=float(oof.tabm_probability.max()), one_prediction_per_participant=True,
        fold_counts={str(k):int(v) for k,v in oof.outer_fold.value_counts().sort_index().items()})
    summary['fit_count'] = int(len(inner)+2*len(folds))
    summary['training_history_epochs'] = len(pd.read_csv(out/'tabm_training_history.csv'))
    summary['fit_seconds'] = float(inner.seconds.sum()+folds.outer_fit_seconds.sum())
    summary['any_inner_epoch_cap_reached'] = bool((inner.epochs_trained==summary['max_epochs']).any())
    summary['any_outer_epoch_cap_reached'] = bool((folds.epochs_trained==summary['max_epochs']).any())
    summary['training_source_hashes'] = json.loads((out/'tabm_source_hashes.json').read_text(encoding='utf-8'))
    summary['final_source_hashes'] = {p:checksum(p) for p in summary['training_source_hashes']}
    (out/'tabm_final_source_hashes.json').write_text(json.dumps(summary['final_source_hashes'], indent=2), encoding='utf-8')
    for protected, expected in summary['protected_hashes_before'].items():
        if checksum(protected) != expected:
            raise ValueError(f'Protected file changed during reporting: {protected}')
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary

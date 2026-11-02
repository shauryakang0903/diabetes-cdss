"""Descriptive raw-input checks; never fit or modify the prediction pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import CORE_FEATURES
from src.data.nhanes_model_preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.models.nhanes_phase2_model import (
    DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, MODEL_VERSION,
    load_metadata, sha256, canonical_json_sha256, validate_input_columns,
)

ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_PATH = ROOT / 'data/processed/nhanes/nhanes_development_2011_2020.csv'
REFERENCE_PATH = ROOT / 'models/nhanes/phase2_input_reference_metadata.json'
REFERENCE_VERSION = 'nhanes_input_reference_v1'
DISCLAIMER = ("This indicator describes similarity to the model's development data. "
              "It is not a clinical confidence score and does not quantify the probability "
              "that the prediction is correct.")


def generate_reference_metadata():
    """Read only the fixed development file, using observed values before imputation."""
    metadata = load_metadata()
    source_hash = sha256(DEVELOPMENT_PATH)
    if source_hash != metadata['development_csv_sha256']:
        raise ValueError('Development cohort checksum differs from frozen provenance')
    frame = pd.read_csv(DEVELOPMENT_PATH)
    data = validate_input_columns(frame)
    if 'seqn' not in frame or frame['seqn'].isna().any() or frame['seqn'].duplicated().any():
        raise ValueError('Development participant identifiers must be present and unique')
    if len(frame) != 25580 or frame['diabetes'].value_counts().to_dict() != {0: 21890, 1: 3690}:
        raise ValueError('Unexpected frozen development cohort size or outcome counts')
    numeric = {}
    for feature in NUMERIC_FEATURES:
        values = data[feature].dropna().to_numpy()
        if not len(values):
            raise ValueError(f'No observed development values: {feature}')
        quantiles = np.quantile(values, [.01, .05, .5, .95, .99], method='linear')
        numeric[feature] = dict(count=int(len(values)), min=float(values.min()),
                               max=float(values.max()), **dict(zip(
                                   ['p01', 'p05', 'median', 'p95', 'p99'], map(float, quantiles))))
    result = dict(reference_version=REFERENCE_VERSION, model_version=MODEL_VERSION,
                  source_development_path=str(DEVELOPMENT_PATH), source_cohort_sha256=source_hash,
                  frozen_model_sha256=sha256(DEFAULT_ARTIFACT_PATH),
                  frozen_metadata_sha256=sha256(DEFAULT_METADATA_PATH), core_features=CORE_FEATURES,
                  frozen_metadata_canonical_sha256=canonical_json_sha256(DEFAULT_METADATA_PATH),
                  numeric_statistics=numeric,
                  categorical_observed_values={f: sorted(data[f].dropna().unique().tolist())
                                               for f in CATEGORICAL_FEATURES},
                  completeness_categories={'Complete': [10, 10], 'Minor missingness': [8, 9],
                                           'Reduced completeness': [0, 7]},
                  applicability_labels={
                      'Within development-data profile': 'Complete; typical numeric and known categorical inputs',
                      'Interpret with caution': 'Minor missingness or uncommon values, without stronger flags',
                      'Limited model applicability': 'Reduced completeness, out-of-range values or unseen categories'},
                  method={'version': 1, 'quantile_method': 'linear', 'numpy_version': np.__version__,
                          'values': 'Raw observed development values only; no imputation or model fitting',
                          'typical': 'p01 <= value <= p99',
                          'uncommon': 'Within inclusive observed min/max, outside p01/p99'},
                  disclaimer=DISCLAIMER,
                  scope='Project-defined interface labels; post-hoc engineering, not validated uncertainty')
    if sha256(DEVELOPMENT_PATH) != source_hash:
        raise ValueError('Development cohort changed during reference generation')
    if list(result['categorical_observed_values'].values()) != metadata['learned_categories']:
        raise ValueError('Observed development categories differ from frozen encoder categories')
    return result


def save_reference_metadata():
    reference = generate_reference_metadata()
    content = json.dumps(reference, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if REFERENCE_PATH.exists() and REFERENCE_PATH.read_text(encoding='utf-8') != content:
        raise ValueError('Reference already exists with different contents; investigate before replacing')
    if not REFERENCE_PATH.exists():
        REFERENCE_PATH.write_text(content, encoding='utf-8')
    return reference


def load_reference_metadata(path=REFERENCE_PATH, *, metadata_path=DEFAULT_METADATA_PATH,
                            artifact_path=DEFAULT_ARTIFACT_PATH):
    """Inference reads saved JSON and frozen provenance, never either cohort."""
    reference = json.loads(Path(path).read_text(encoding='utf-8'))
    if not reference.get('frozen_metadata_canonical_sha256'):
        raise ValueError('Input reference missing frozen_metadata_canonical_sha256. '
                         'Update the committed phase2_input_reference_metadata.json from the verified '
                         'frozen metadata; do not regenerate the predictive model or skip provenance checks.')
    metadata = load_metadata(metadata_path)
    if (reference.get('reference_version') != REFERENCE_VERSION
            or reference.get('model_version') != MODEL_VERSION
            or reference.get('model_version') != metadata['model_version']
            or reference.get('core_features') != CORE_FEATURES
            or reference.get('core_features') != metadata['core_features']
            or reference.get('source_cohort_sha256') != metadata['development_csv_sha256']
            or reference.get('frozen_model_sha256') != sha256(artifact_path)
            or reference.get('frozen_model_sha256') != metadata['artifact_sha256']
            or reference['frozen_metadata_canonical_sha256'] != canonical_json_sha256(metadata_path)):
        raise ValueError('Input reference does not match the frozen model/development provenance')
    if set(reference['numeric_statistics']) != set(NUMERIC_FEATURES) or set(
            reference['categorical_observed_values']) != set(CATEGORICAL_FEATURES):
        raise ValueError('Input reference features differ from Core-10 specification')
    for feature, stats in reference['numeric_statistics'].items():
        ordered = [stats[k] for k in ['min', 'p01', 'p05', 'median', 'p95', 'p99', 'max']]
        if not np.isfinite(ordered).all() or ordered != sorted(ordered) or stats['count'] <= 0:
            raise ValueError(f'Invalid reference statistics: {feature}')
    if [reference['categorical_observed_values'][f] for f in CATEGORICAL_FEATURES] != metadata['learned_categories']:
        raise ValueError('Input reference categories differ from frozen development categories')
    return reference


def numeric_status(value, stats):
    if pd.isna(value):
        return 'missing'
    if value < stats['min'] or value > stats['max']:
        return 'out_of_development_range'
    return 'typical' if stats['p01'] <= value <= stats['p99'] else 'uncommon'


def categorical_status(value, observed):
    if pd.isna(value):
        return 'missing'
    return 'known_category' if value in observed else 'unseen_development_category'


def assess_input_quality(patient, reference=None):
    reference = load_reference_metadata() if reference is None else reference
    frame = patient if isinstance(patient, pd.DataFrame) else pd.DataFrame([patient])
    if len(frame) != 1:
        raise ValueError('Input quality requires exactly one patient')
    if any(isinstance(v, (bool, np.bool_, complex)) for k, v in frame.iloc[0].items() if k in CORE_FEATURES):
        raise ValueError('Use numeric NHANES codes rather than boolean values')
    data = validate_input_columns(frame)
    statuses = {}
    for feature in CORE_FEATURES:
        value = data.iloc[0][feature]
        status = (numeric_status(value, reference['numeric_statistics'][feature]) if feature in NUMERIC_FEATURES
                  else categorical_status(value, reference['categorical_observed_values'][feature]))
        statuses[feature] = {'value': None if pd.isna(value) else float(value), 'status': status}
    flagged = lambda status: [f for f in CORE_FEATURES if statuses[f]['status'] == status]
    missing, uncommon = flagged('missing'), flagged('uncommon')
    outside, unseen = flagged('out_of_development_range'), flagged('unseen_development_category')
    observed = 10 - len(missing)
    completeness = 'Complete' if observed == 10 else 'Minor missingness' if observed >= 8 else 'Reduced completeness'
    if observed <= 7 or outside or unseen:
        overall = 'Limited model applicability'
        note = ('Prediction was generated, but completeness is reduced, an input is outside the observed '
                'development range, or a category was unseen. Model applicability may therefore be limited.')
    elif missing or uncommon:
        overall = 'Interpret with caution'
        note = 'The prediction was generated, but interpretation should account for missing or uncommon input values.'
    else:
        overall = 'Within development-data profile'
        note = "All supplied inputs are complete and fall within the model's typical development-data range."
    return dict(expected_features=10, observed_features=observed, missing_features=len(missing),
                missing_feature_names=missing, completeness_percent=float(observed * 10),
                completeness_label=completeness, uncommon_features=uncommon,
                out_of_range_features=outside, unseen_categories=unseen,
                overall_applicability=overall, applicability_note=note, feature_status=statuses,
                reference_version=reference['reference_version'], disclaimer=DISCLAIMER)

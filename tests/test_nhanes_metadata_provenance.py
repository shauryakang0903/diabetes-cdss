"""Formatting-independent JSON provenance with strict binary/semantic checks."""
import json
from pathlib import Path

import pytest

from src.decision_support import nhanes_input_quality as quality
from src.models.nhanes_phase2_model import (
    DEFAULT_METADATA_PATH, DEFAULT_ARTIFACT_PATH, canonical_json_sha256, sha256,
)


@pytest.fixture
def originals():
    return (json.loads(DEFAULT_METADATA_PATH.read_text(encoding='utf-8')),
            json.loads(quality.REFERENCE_PATH.read_text(encoding='utf-8')))


def write_json(path, data, *, crlf=False, compact=False):
    text = json.dumps(data, indent=None if compact else 4, sort_keys=compact, ensure_ascii=compact,
                       separators=(',', ':') if compact else None, allow_nan=False) + '\n'
    path.write_bytes(text.replace('\n', '\r\n').encode('utf-8') if crlf else text.encode('utf-8'))
    return path


def test_canonical_json_ignores_line_endings_indentation_key_order_and_escaping(tmp_path):
    data = {'z': ['café', 2], 'a': {'value': True}}
    lf = write_json(tmp_path/'lf.json', data)
    crlf = write_json(tmp_path/'crlf.json', data, crlf=True)
    compact = write_json(tmp_path/'compact.json', data, compact=True)
    assert sha256(lf) != sha256(crlf) != sha256(compact)
    assert canonical_json_sha256(lf) == canonical_json_sha256(crlf) == canonical_json_sha256(compact)


@pytest.mark.parametrize('field,value', [('model_version', 'changed'), ('artifact_sha256', '0'*64)])
def test_changed_metadata_values_change_canonical_hash(tmp_path, originals, field, value):
    metadata, _ = originals
    before = write_json(tmp_path/'before.json', metadata)
    metadata[field] = value
    after = write_json(tmp_path/'after.json', metadata)
    assert canonical_json_sha256(before) != canonical_json_sha256(after)


@pytest.mark.parametrize('crlf,compact', [(False, False), (True, False), (False, True)])
def test_reference_accepts_logically_identical_metadata(tmp_path, originals, crlf, compact):
    metadata, _ = originals
    path = write_json(tmp_path/'metadata.json', metadata, crlf=crlf, compact=compact)
    result = quality.load_reference_metadata(metadata_path=path)
    assert result['frozen_metadata_canonical_sha256'] == canonical_json_sha256(path)
    assert result['frozen_model_sha256'] == sha256(DEFAULT_ARTIFACT_PATH)


def test_wrong_binary_model_still_fails(tmp_path):
    artifact = tmp_path/'wrong.joblib'
    artifact.write_bytes(DEFAULT_ARTIFACT_PATH.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='provenance'):
        quality.load_reference_metadata(artifact_path=artifact)


@pytest.mark.parametrize('field,value', [
    ('model_version', 'other-model'), ('development_csv_sha256', '0'*64),
    ('core_features', ['age']), ('artifact_sha256', '0'*64),
])
def test_semantic_mismatch_rejected_even_with_matching_new_canonical_hash(tmp_path, originals, field, value):
    metadata, reference = originals
    metadata[field] = value
    path = write_json(tmp_path/'metadata.json', metadata)
    # Re-pinning the modified JSON alone must not defeat cross-document checks.
    reference['frozen_metadata_canonical_sha256'] = canonical_json_sha256(path)
    ref_path = write_json(tmp_path/'reference.json', reference)
    with pytest.raises(ValueError):
        quality.load_reference_metadata(ref_path, metadata_path=path)


def test_unrelated_real_metadata_change_also_fails_full_content_pin(tmp_path, originals):
    metadata, _ = originals
    metadata['project_purpose'] = 'changed purpose'
    with pytest.raises(ValueError, match='provenance'):
        quality.load_reference_metadata(metadata_path=write_json(tmp_path/'metadata.json', metadata))


def test_old_reference_requires_explicit_migration(tmp_path, originals):
    _, reference = originals
    reference.pop('frozen_metadata_canonical_sha256')
    path = write_json(tmp_path/'legacy.json', reference)
    with pytest.raises(ValueError, match='missing frozen_metadata_canonical_sha256.*Update the committed'):
        quality.load_reference_metadata(path)


def test_nonfinite_json_is_not_canonicalized(tmp_path):
    path = tmp_path/'invalid.json'
    path.write_text('{"value": NaN}', encoding='utf-8')
    with pytest.raises(ValueError):
        canonical_json_sha256(path)


def test_full_backend_initializes_with_linux_metadata_and_preserves_probability(tmp_path, originals):
    from scripts.demo_nhanes_cdss import synthetic_profiles
    from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
    metadata, _ = originals
    path = write_json(tmp_path/'metadata.json', metadata)
    result = NhanesDiabetesCDSS(metadata_path=path).predict(
        synthetic_profiles()['relatively_lower_scoring_example'])
    assert result['predicted_probability'] == pytest.approx(.005870712921023369, abs=1e-8)
    assert result['model_derived_risk']['level'] == 'LOW'
    assert result['explanation_integrity']['probability_check_passed']
    assert len(result['all_feature_contributions']) == 10 and len(result['recommendations']) == 4
    assert result['input_quality']['observed_features'] == 10


def test_binary_and_original_metadata_semantics_remain_unchanged():
    assert sha256(DEFAULT_ARTIFACT_PATH) == 'fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528'
    assert canonical_json_sha256(DEFAULT_METADATA_PATH) == 'cacd39ae02b71aa42ab1dc1133687a22c7e36bfcfabbbbefd17482674108cf79'

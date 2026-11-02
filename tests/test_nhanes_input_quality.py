"""Development reference checks and strict regression of the existing outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.decision_support import nhanes_input_quality as quality
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, sha256
from scripts.demo_nhanes_cdss import synthetic_profiles


@pytest.fixture(scope='module')
def reference():
    return quality.load_reference_metadata()


@pytest.fixture(scope='module')
def service():
    return NhanesDiabetesCDSS()


@pytest.mark.parametrize('missing, label', [(0, 'Complete'), (1, 'Minor missingness'),
                                         (2, 'Minor missingness'), (3, 'Reduced completeness'),
                                         (10, 'Reduced completeness')])
def test_completeness(reference, missing, label):
    patient = synthetic_profiles()['intermediate_scoring_example'].copy()
    for feature in quality.CORE_FEATURES[:missing]:
        patient[feature] = None
    result = quality.assess_input_quality(patient, reference)
    assert result['observed_features'] == 10 - missing
    assert result['missing_features'] == missing
    assert result['completeness_percent'] == (10 - missing) * 10
    assert result['completeness_label'] == label
    assert list(result['feature_status']) == quality.CORE_FEATURES
    assert result['disclaimer'] == quality.DISCLAIMER
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('value, expected', [(1, 'typical'), (99, 'typical'),
    (0, 'uncommon'), (.5, 'uncommon'), (99.5, 'uncommon'), (100, 'uncommon'),
    (-1, 'out_of_development_range'), (101, 'out_of_development_range'), (np.nan, 'missing')])
def test_numeric_boundaries(value, expected):
    assert quality.numeric_status(value, dict(min=0, p01=1, p99=99, max=100)) == expected


@pytest.mark.parametrize('value, expected', [(1, 'known_category'), (99, 'unseen_development_category'),
                                           (None, 'missing')])
def test_categories(value, expected):
    assert quality.categorical_status(value, [1, 2]) == expected


def test_generation_deterministic_development_only(monkeypatch, reference):
    read = pd.read_csv
    reads = []
    before = sha256(quality.DEVELOPMENT_PATH)
    def guarded(path, *args, **kwargs):
        assert Path(path).resolve() == quality.DEVELOPMENT_PATH.resolve()
        reads.append(path)
        return read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', guarded)
    assert quality.generate_reference_metadata() == quality.generate_reference_metadata() == reference
    assert len(reads) == 2
    assert sha256(quality.DEVELOPMENT_PATH) == before


def test_all_legacy_outputs_identical(service):
    fixture = json.loads((Path(__file__).parent / 'fixtures/nhanes_input_quality_pre_layer_baseline.json').read_text())
    for scenario in fixture['scenarios'].values():
        result = service.predict(scenario['patient'])
        assert {k: v for k, v in result.items() if k != 'input_quality'} == scenario['result']
        quality.assess_input_quality(scenario['patient'], service.input_reference)
        assert service.predict(scenario['patient']) == result
    assert sha256(DEFAULT_ARTIFACT_PATH) == fixture['artifact_sha256'] == 'fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528'
    assert sha256(DEFAULT_METADATA_PATH) == fixture['metadata_sha256'] == '57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3'


def test_inference_no_cohort_reads_and_unknown_category_warning(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Inference must not read a cohort')
    monkeypatch.setattr(pd, 'read_csv', forbidden)
    backend = NhanesDiabetesCDSS()
    patient = synthetic_profiles()['intermediate_scoring_example'].copy()
    patient['sex'] = 99
    result = backend.predict(patient)
    assert 0 <= result['predicted_probability'] <= 1
    assert result['input_quality']['unseen_categories'] == ['sex']
    assert result['input_quality']['overall_applicability'] == 'Limited model applicability'


def test_overall_labels(reference):
    patient = synthetic_profiles()['intermediate_scoring_example'].copy()
    assert quality.assess_input_quality(patient, reference)['overall_applicability'] == 'Within development-data profile'
    patient['hdl_mg_dl'] = None
    assert quality.assess_input_quality(patient, reference)['overall_applicability'] == 'Interpret with caution'
    patient['bmi'] = reference['numeric_statistics']['bmi']['max'] + 1
    assert quality.assess_input_quality(patient, reference)['overall_applicability'] == 'Limited model applicability'


def test_bad_reference_fails(tmp_path, reference):
    bad = dict(reference, source_cohort_sha256='incorrect')
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='provenance'):
        quality.load_reference_metadata(path)


def test_streamlit_quality_workflow():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(quality.ROOT / 'app.py'), default_timeout=60).run()
    for profile in ['relatively_lower_scoring_example', 'higher_scoring_example']:
        app.selectbox(key='demo_profile').select(profile).run()
        app.button[0].click().run()
        assert not app.exception and not app.error
        assert 'Input Data Quality' in [x.value for x in app.subheader]
        result = app.session_state['assessment']
        assert result['input_quality']['observed_features'] == 10
        assert result['model_derived_risk']['label']
        assert len(app.tabs) == 4
        assert len(result['all_feature_contributions']) == 10
    app.selectbox(key='demo_profile').select('intermediate_scoring_example').run()
    app.multiselect(key='unavailable_measurements').select('hdl_mg_dl').run()
    # Form values are sent together by the submit button, as in the browser.
    app.number_input(key='bmi').set_value(58.)
    app.button[0].click().run()
    assert not app.exception and not app.error
    result = app.session_state['assessment']
    assert result['input_quality']['observed_features'] == 9
    assert result['input_quality']['uncommon_features'] == ['bmi']
    assert result['input_quality']['completeness_label'] == 'Minor missingness'

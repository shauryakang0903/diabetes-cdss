"""Synthetic inference/deployment checks: no cohort loading or model fitting."""
import json
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from scripts import check_streamlit_deployment as deployment
from src.models import nhanes_phase2_model as frozen

ROOT = Path(__file__).resolve().parents[1]


def deployment_copy(destination):
    """A repository-like copy with no data, research results, tests or DL modules."""
    files = [ROOT/name for name in deployment.REQUIRED_FILES]
    files += [p for p in (ROOT/'src').rglob('*.py') if 'tabm' not in p.name]
    files += [ROOT/'scripts/demo_nhanes_cdss.py', ROOT/'scripts/check_streamlit_deployment.py']
    for source in files:
        target = destination/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def test_runtime_requirements_pin_frozen_versions_without_research_packages():
    requirements = [Requirement(line) for line in (ROOT/'requirements.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
    linux = {r.name: r for r in requirements if r.marker is None or r.marker.evaluate({'platform_system':'Linux'})}
    windows = {r.name: r for r in requirements if r.marker is None or r.marker.evaluate({'platform_system':'Windows'})}
    assert 'xgboost-cpu' in linux and 'xgboost' not in linux
    assert 'xgboost' in windows and 'xgboost-cpu' not in windows
    assert not {'torch','tabm','pytest','lime','jupyter'} & set(linux)
    metadata = frozen.load_metadata()
    for name, expected in metadata['package_versions'].items():
        assert expected in linux['xgboost-cpu' if name == 'xgboost' else name].specifier


def test_official_cpu_distribution_version_fallback(monkeypatch):
    def version(name):
        if name == 'xgboost':
            raise PackageNotFoundError(name)
        if name == 'xgboost-cpu':
            return '3.4.1'
        raise PackageNotFoundError(name)
    monkeypatch.setattr(frozen, 'version', version)
    assert frozen.runtime_package_version('xgboost') == '3.4.1'
    with pytest.raises(PackageNotFoundError):
        frozen.runtime_package_version('missing-package')


def test_health_check_in_unrelated_cwd_without_fitting(tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    def forbidden(*args, **kwargs):
        pytest.fail('Deployment must never fit a model')
    monkeypatch.setattr(Pipeline, 'fit', forbidden)
    monkeypatch.setattr(XGBClassifier, 'fit', forbidden)
    monkeypatch.chdir(tmp_path)
    report = deployment.check_deployment()
    assert report['passed'] and report['cohort_csv_reads'] == 0
    assert report['probability'] == pytest.approx(.005870712921023369, abs=1e-8)
    assert report['model_derived_stratum'] == 'LOW'
    assert report['model_sha256'] == deployment.EXPECTED_MODEL_SHA256
    assert report['metadata_sha256'] == frozen.sha256(frozen.DEFAULT_METADATA_PATH)
    assert report['metadata_canonical_sha256'] == deployment.EXPECTED_METADATA_CANONICAL_SHA256


def test_missing_essential_artifacts_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError, match='Required deployment files missing:.*phase2_xgboost_pipeline'):
        deployment.check_deployment()


def test_minimal_repository_copy_has_no_dataset_or_result_dependency(tmp_path):
    bundle = deployment_copy(tmp_path/'public-copy')
    # Simulate Git's Linux checkout, including the metadata raw-byte hash drift.
    metadata = bundle/'models/nhanes/phase2_xgboost_metadata.json'
    metadata.write_bytes(metadata.read_bytes().replace(b'\r\n', b'\n'))
    assert not (bundle/'data').exists() and not (bundle/'results').exists()
    assert not (bundle/'src/models/nhanes_tabm.py').exists()
    result = subprocess.run([sys.executable, str(bundle/'scripts/check_streamlit_deployment.py')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['passed'] and not report['optional_global_shap_image']
    assert report['metadata_sha256'] != deployment.EXPECTED_METADATA_SHA256
    assert report['metadata_canonical_sha256'] == deployment.EXPECTED_METADATA_CANONICAL_SHA256
    assert report['xgboost_module_version'] == '3.4.1'


def test_existing_synthetic_outputs_preserved():
    from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
    fixture = json.loads((ROOT/'tests/fixtures/nhanes_input_quality_pre_layer_baseline.json').read_text())
    service = NhanesDiabetesCDSS()
    for case in fixture['scenarios'].values():
        result = service.predict(case['patient'])
        assert {k:v for k,v in result.items() if k != 'input_quality'} == case['result']


def test_streamlit_form_and_all_decision_support_sections():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
    assert not app.exception and len(app.number_input) == 7 and len(app.selectbox) == 4
    app.selectbox(key='demo_profile').select('relatively_lower_scoring_example').run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    result = app.session_state['assessment']
    assert result['model_derived_risk']['level'] == 'LOW'
    assert result['input_quality']['observed_features'] == 10
    assert len(result['all_feature_contributions']) == 10 and len(app.tabs) == 4
    assert 'Input Data Quality' in [section.value for section in app.subheader]

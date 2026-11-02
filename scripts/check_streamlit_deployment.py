"""Inference-only deployment preflight; independent of cwd, cohorts and test tools."""
import json
import math
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_FILES = (
    'app.py', 'requirements.txt', '.streamlit/config.toml',
    'models/nhanes/phase2_xgboost_pipeline.joblib',
    'models/nhanes/phase2_xgboost_metadata.json',
    'models/nhanes/phase2_input_reference_metadata.json',
    'configs/nhanes_recommendation_rules.json',
)
EXPECTED_MODEL_SHA256 = 'fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528'
EXPECTED_METADATA_SHA256 = '57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3'
# The byte hash above is historical provenance only; Git can normalize JSON text.
EXPECTED_METADATA_CANONICAL_SHA256 = 'cacd39ae02b71aa42ab1dc1133687a22c7e36bfcfabbbbefd17482674108cf79'
EXPECTED_LOWER_PROBABILITY = 0.005870712921023369
RUNTIME_PACKAGES = ('streamlit', 'numpy', 'pandas', 'scikit-learn', 'xgboost',
                    'joblib', 'scipy', 'shap', 'matplotlib', 'pillow')


def check_deployment():
    missing = [name for name in REQUIRED_FILES if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError('Required deployment files missing: ' + ', '.join(missing))
    from src.models.nhanes_phase2_model import sha256, canonical_json_sha256, runtime_package_version
    if sha256(ROOT/REQUIRED_FILES[3]) != EXPECTED_MODEL_SHA256:
        raise ValueError('Frozen model checksum mismatch; do not regenerate the model')
    if canonical_json_sha256(ROOT/REQUIRED_FILES[4]) != EXPECTED_METADATA_CANONICAL_SHA256:
        raise ValueError('Frozen metadata checksum mismatch; restore the verified original')
    import xgboost
    if xgboost.__version__ != runtime_package_version('xgboost'):
        raise ValueError('Imported XGBoost module differs from its distribution version')
    from scripts.demo_nhanes_cdss import synthetic_profiles
    from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
    # A public inference deployment must not need any cohort or research CSV.
    with patch('pandas.read_csv', side_effect=AssertionError('Deployment inference attempted a CSV read')):
        service = NhanesDiabetesCDSS()
        result = service.predict(synthetic_profiles()['relatively_lower_scoring_example'])
    probability = result['predicted_probability']
    if not math.isfinite(probability) or not 0 <= probability <= 1 or not math.isclose(
            probability, EXPECTED_LOWER_PROBABILITY, rel_tol=0, abs_tol=1e-8):
        raise ValueError('Synthetic regression probability differs from the frozen reference')
    if (result['model_derived_risk']['level'] != 'LOW'
            or not result['explanation_integrity']['probability_check_passed']
            or len(result['all_feature_contributions']) != 10
            or result['input_quality']['observed_features'] != 10
            or len(result['recommendations']) != 4):
        raise ValueError('Decision-support output integrity check failed')
    return dict(passed=True, python=platform.python_version(),
                versions={name:runtime_package_version(name) for name in RUNTIME_PACKAGES},
                model_sha256=EXPECTED_MODEL_SHA256, metadata_sha256=sha256(ROOT/REQUIRED_FILES[4]),
                metadata_canonical_sha256=EXPECTED_METADATA_CANONICAL_SHA256,
                historical_metadata_sha256=EXPECTED_METADATA_SHA256,
                xgboost_module_version=xgboost.__version__,
                probability=probability, probability_percent=100*probability,
                model_derived_stratum=result['model_derived_risk']['level'],
                shap_integrity=True, recommendations=True, input_quality=True,
                required_files=list(REQUIRED_FILES), cohort_csv_reads=0,
                optional_global_shap_image=(ROOT/'results/nhanes/xai/figures/shap_global_core10_bar.png').is_file())


if __name__ == '__main__':
    try:
        print(json.dumps(check_deployment(), indent=2, allow_nan=False))
    except Exception as error:
        print(f'Deployment check FAILED: {error}', file=sys.stderr)
        raise SystemExit(1) from error

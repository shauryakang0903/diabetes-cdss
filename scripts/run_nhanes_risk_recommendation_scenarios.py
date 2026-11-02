"""Audit existing synthetic profiles through the post-prediction framework."""

import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.demo_nhanes_cdss import synthetic_profiles
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.decision_support.nhanes_recommendations import CONFIG_PATH, SECTIONS
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/nhanes/cdss"
BASELINE_PATH = ROOT / "tests/fixtures/nhanes_cdss_pre_recommendation_baseline.json"


def run(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if any(output_dir.resolve().is_relative_to((ROOT / folder).resolve()) for folder in ("data", "models")):
        raise ValueError("Scenario output must not be saved under frozen data or models")
    inputs = {"artifact": DEFAULT_ARTIFACT_PATH, "metadata": DEFAULT_METADATA_PATH, "rules": CONFIG_PATH,
              "pre_framework_baseline": BASELINE_PATH}
    before = {key: sha256(path) for key, path in inputs.items()}
    protected = {str(p): sha256(p) for p in (ROOT/"results/nhanes/xai").rglob("*") if p.is_file()}
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline["artifact_sha256"] != before["artifact"]:
        raise ValueError("Pre-framework baseline belongs to a different frozen artifact")
    service = NhanesDiabetesCDSS()
    rows, details = [], {}
    for name, patient in synthetic_profiles().items():
        result = service.predict(patient)
        old = baseline["profiles"][name]
        if patient != old["patient"] or any(result[key] != value for key, value in old["result"].items()):
            raise AssertionError("Pre-existing probability, SHAP or other backend output changed")
        counts = {section: len(result["recommendations"][section]) for section in SECTIONS}
        rows.append(dict(scenario=name, probability=result["predicted_probability"],
                         risk_stratum=result["model_derived_risk"]["level"],
                         triggered_rule_ids=json.dumps(result["recommendation_provenance"]["triggered_rule_ids"]),
                         clinical_followup_count=counts["clinical_follow_up"], monitoring_count=counts["monitoring"],
                         lifestyle_count=counts["lifestyle"], feature_aware_count=counts["feature_aware"]))
        details[name] = {"inputs": patient, "model_derived_risk": result["model_derived_risk"],
                         "recommendations": result["recommendations"], "legacy_result_identical": True}
    after = {key: sha256(path) for key, path in inputs.items()}
    if before != after or protected != {path: sha256(path) for path in protected}:
        raise AssertionError("Protected frozen files or previous explanation results changed")
    provenance = dict(framework_version=service.recommendation_rules["framework_version"],
                      model_version=service.metadata["model_version"], completed_utc=datetime.now(timezone.utc).isoformat(),
                      input_paths={key: str(path) for key, path in inputs.items()}, sha256_before=before, sha256_after=after,
                      protected_xai_results_sha256=protected,
                      exact_thresholds=service.metadata["reference_thresholds"],
                      sources=service.recommendation_rules["sources"], scenarios=details,
                      frozen_predictions_identical=True, shap_outputs_identical=True,
                      model_retrained=False, preprocessing_refit=False, thresholds_optimized=False,
                      temporal_cohort_used=False, cohort_data_loaded=False,
                      moderate_scenario_note="No synthetic profile was altered to force MODERATE; exact moderate boundaries are covered by unit tests.",
                      python_packages={key: version(key) for key in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "streamlit")},
                      source_sha256={str(p): sha256(p) for p in (Path(__file__),
                          ROOT/"src/decision_support/nhanes_recommendations.py", ROOT/"src/decision_support/nhanes_diabetes_cdss.py")})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir/"risk_recommendation_scenarios.csv", index=False)
    (output_dir/"risk_recommendation_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(frame.to_string(index=False))
    return frame, provenance


if __name__ == "__main__":
    run()

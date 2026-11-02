"""Final audit only: synthetic inference and saved experimental evidence."""

import copy
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from scripts.demo_nhanes_cdss import synthetic_profiles
from src.data.preprocessing import CORE_FEATURES
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.decision_support.nhanes_recommendations import CONFIG_PATH, SECTIONS, add_recommendation_layer, model_derived_risk
from src.decision_support.prototype_presentation import assess_patient, explanation_table, probability_text, reference_table
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, predict_probabilities, sha256

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/nhanes"
OUTPUT_DIR = RESULTS / "cdss"
BASELINE = ROOT / "tests/fixtures/nhanes_cdss_pre_recommendation_baseline.json"
HISTORY = {
    "baseline": RESULTS / "baseline_model_comparison_summary.csv",
    "nested": RESULTS / "nested_tuning_summary.csv",
    "temporal": RESULTS / "temporal_validation_metrics.csv",
    "transport": RESULTS / "temporal_validation_thresholds.csv",
    "missingness": RESULTS / "temporal_validation_missingness_robustness.csv",
    "thresholds": RESULTS / "threshold_candidate_points.csv",
    "global_shap": RESULTS / "xai/shap_global_provenance.json",
    "lime_agreement": RESULTS / "xai/shap_lime_agreement.csv",
    "lime_explanations": RESULTS / "xai/lime_synthetic_explanations.csv",
    "lime_provenance": RESULTS / "xai/lime_provenance.json",
    "risk_provenance": OUTPUT_DIR / "risk_recommendation_provenance.json",
}
PROTECTED = [DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, CONFIG_PATH, BASELINE,
             ROOT/"app.py", ROOT/"src/decision_support/nhanes_diabetes_cdss.py",
             ROOT/"src/decision_support/nhanes_recommendations.py",
             ROOT/"src/decision_support/prototype_presentation.py",
             ROOT/"src/models/nhanes_phase2_model.py",
             ROOT/"src/explainability/nhanes_shap_global.py",
             ROOT/"src/explainability/nhanes_lime_local.py", *HISTORY.values()]


def snapshot_inputs():
    return {str(path.relative_to(ROOT)): sha256(path) for path in PROTECTED}


def validation_scenarios():
    profiles = synthetic_profiles()
    # One manually specified software demonstration, not an input search:
    # the existing intermediate profile at age 60, all other inputs unchanged.
    moderate = profiles["intermediate_scoring_example"].copy()
    moderate["age"] = 60
    profiles["moderate_stratum_software_demonstration"] = moderate
    return profiles


def presentation_snapshot(result):
    return dict(probability=probability_text(result), model_version=result["model_version"],
                reference_interpretation=result["reference_interpretation"],
                risk=copy.deepcopy(result["model_derived_risk"]),
                explanations=explanation_table(result).to_dict("records"),
                references=reference_table(result).to_dict("records"),
                recommendations=copy.deepcopy(result["recommendations"]),
                limitations=list(result["limitations"]))


def validate_scenarios(service=None, repetitions=3):
    if repetitions < 2:
        raise ValueError("At least two repetitions are required to check determinism")
    service = NhanesDiabetesCDSS() if service is None else service
    legacy = json.loads(BASELINE.read_text(encoding="utf-8"))["profiles"]
    rows, full = [], {}
    for name, patient in validation_scenarios().items():
        outputs = [assess_patient(patient, service) for _ in range(repetitions)]
        first = outputs[0]
        canonical = [json.dumps(result, sort_keys=True, allow_nan=False) for result in outputs]
        presentations = [json.dumps(presentation_snapshot(r), sort_keys=True, allow_nan=False) for r in outputs]
        assert len(set(canonical)) == len(set(presentations)) == 1, f"Nondeterminism: {name}"
        direct = float(predict_probabilities(pd.DataFrame([patient]), pipeline=service.pipeline)[0])
        assert first["predicted_probability"] == direct, f"Probability changed: {name}"
        assert first["model_derived_risk"] == model_derived_risk(direct, service.metadata, service.recommendation_rules)
        assert set(first["recommendations"]) == set(SECTIONS)
        unlayered = {k: v for k, v in first.items() if k not in ("model_derived_risk", "recommendations", "recommendation_provenance")}
        cloned = copy.deepcopy(unlayered)
        again = add_recommendation_layer(cloned, service.metadata, service.recommendation_rules)
        assert cloned == unlayered and again == first, "Recommendation layer mutated a lower-layer result"
        legacy_equal = None
        if name in legacy:
            legacy_equal = all(first[key] == value for key, value in legacy[name]["result"].items())
            assert legacy_equal, f"Pre-framework prediction/SHAP changed: {name}"
        integrity = first["explanation_integrity"]
        assert integrity["margin_additivity"]["passed"] and integrity["probability_check_passed"]
        rows.append(dict(
            scenario=name, patient_inputs=json.dumps(patient), predicted_probability=direct,
            model_derived_stratum=first["model_derived_risk"]["level"],
            top3_increasing_shap_features=json.dumps([r["feature"] for r in first["top_increasing_contributors"]]),
            top3_decreasing_shap_features=json.dumps([r["feature"] for r in first["top_decreasing_contributors"]]),
            recommendation_rule_ids=json.dumps(first["recommendation_provenance"]["triggered_rule_ids"]),
            clinical_followup_count=len(first["recommendations"]["clinical_follow_up"]),
            monitoring_count=len(first["recommendations"]["monitoring"]), lifestyle_count=len(first["recommendations"]["lifestyle"]),
            feature_aware_count=len(first["recommendations"]["feature_aware"]),
            explanation_additivity_error=integrity["margin_additivity"]["maximum_absolute_error"],
            reconstructed_probability_error=integrity["probability_absolute_error"],
            model_version=first["model_version"], repetitions=repetitions, backend_deterministic=True,
            presentation_deterministic=True, frozen_probability_identical=True,
            legacy_output_identical=legacy_equal, recommendation_layer_preserves_output=True))
        full[name] = first
    assert {r["model_derived_stratum"] for r in rows} == {"LOW", "MODERATE", "HIGH"}
    return pd.DataFrame(rows), full


def verify_integrity(service):
    metadata, prep = service.metadata, service.preprocessor
    shap_provenance = json.loads(HISTORY["global_shap"].read_text(encoding="utf-8"))
    risk_provenance = json.loads(HISTORY["risk_provenance"].read_text(encoding="utf-8"))
    artifact, meta_hash = sha256(DEFAULT_ARTIFACT_PATH), sha256(DEFAULT_METADATA_PATH)
    assert artifact == metadata["artifact_sha256"] == shap_provenance["artifact_sha256"]
    assert meta_hash == shap_provenance["metadata_sha256"] == risk_provenance["sha256_after"]["metadata"]
    assert list(service.pipeline.feature_names_in_) == metadata["core_features"] == CORE_FEATURES
    actual = service.estimator.get_params()
    normalized = {key: "NaN" if isinstance(value, float) and np.isnan(value) else value for key, value in actual.items()}
    assert normalized == metadata["xgboost_hyperparameters"], "Frozen hyperparameters changed"
    transformers = {name: (obj, list(columns)) for name, obj, columns in prep.transformers_ if name != "remainder"}
    numeric, num_columns = transformers["numeric"]
    categorical, cat_columns = transformers["categorical"]
    spec = metadata["preprocessing_specification"]
    assert num_columns == spec["numeric_features"] and cat_columns == spec["categorical_features"]
    assert list(numeric.named_steps) == ["imputer"] and spec["numeric_scaling"] == "none"
    assert list(categorical.named_steps) == ["imputer", "encoder"]
    assert numeric.named_steps["imputer"].strategy == spec["numeric_imputation"] == "median"
    assert categorical.named_steps["imputer"].strategy == spec["categorical_imputation"] == "most_frequent"
    assert numeric.named_steps["imputer"].keep_empty_features and categorical.named_steps["imputer"].keep_empty_features
    encoder = categorical.named_steps["encoder"]
    assert encoder.handle_unknown == "ignore" and not encoder.sparse_output and encoder.drop is None
    assert encoder.min_frequency is None and encoder.max_categories is None
    assert numeric.named_steps["imputer"].statistics_.tolist() == metadata["learned_numeric_medians"]
    assert categorical.named_steps["imputer"].statistics_.tolist() == metadata["learned_categorical_modes"]
    assert [v.tolist() for v in encoder.categories_] == metadata["learned_categories"]
    thresholds = pd.read_csv(HISTORY["thresholds"], float_precision="round_trip")
    points = thresholds[thresholds.model == "XGBoost"].set_index("operating_point")
    names = {"default": "default_0.50", "sensitivity_80_reference": "nearest_sensitivity_0.80",
             "sensitivity_90_reference": "nearest_sensitivity_0.90", "youden_reference": "youden_j_maximum"}
    assert {key: float(points.loc[point, "threshold"]) for key, point in names.items()} == metadata["reference_thresholds"]
    return dict(passed=True, artifact_sha256=artifact, metadata_sha256=meta_hash,
                model_version=metadata["model_version"], core10_features=CORE_FEATURES,
                xgboost_hyperparameters=normalized, preprocessing_specification=spec,
                learned_numeric_medians=metadata["learned_numeric_medians"],
                learned_categorical_modes=metadata["learned_categorical_modes"], learned_categories=metadata["learned_categories"],
                reference_thresholds=metadata["reference_thresholds"], comparison="exact match to frozen metadata and prior provenance")


def historical_summary():
    """Read saved result tables only; never rerun predictive experiments."""
    def table(name):
        return pd.read_csv(HISTORY[name], float_precision="round_trip")
    nested, temporal, transport, missing = [table(key) for key in ("nested", "temporal", "transport", "missingness")]
    n = nested[nested.model == "XGBoost"].iloc[0]
    t = temporal[temporal.model == "XGBoost"].iloc[0]
    references = transport[transport.model == "XGBoost"]
    thresholds = {}
    for key in ("0.80", "0.90"):
        row = references[references.operating_point == "nearest_sensitivity_" + key].iloc[0]
        thresholds[key] = {field: float(row[field]) for field in ("threshold", "sensitivity", "specificity")}
    shap = json.loads(HISTORY["global_shap"].read_text(encoding="utf-8"))
    agreements = table("lime_agreement").to_dict("records")
    fidelity = table("lime_explanations").groupby("profile", sort=False)[
        ["local_weighted_r2", "surrogate_prediction", "surrogate_absolute_error", "predicted_probability"]].first()
    return dict(
        evidence_scope="Previously saved experiments only; no rerun, no cohort loading",
        development=dict(cohort="NHANES 2011–March 2020", n=int(t.development_n), nested_cv_outer_folds=int(n.outer_folds), nested_cv_inner_folds=int(n.inner_folds),
                         **{m: {"mean": float(n[m+"_mean"]), "sd": float(n[m+"_std"])} for m in ("roc_auc", "average_precision", "brier_score")}),
        temporal=dict(cohort="NHANES 2021–2023", n=int(t.temporal_n),
                      **{m: float(t[m]) for m in ("roc_auc", "average_precision", "brier_score", "log_loss")},
                      change_vs_nested_roc_auc=float(t.roc_auc-n.roc_auc_mean),
                      change_vs_nested_average_precision=float(t.average_precision-n.average_precision_mean),
                      missingness_groups=missing[missing.model == "XGBoost"].to_dict("records")),
        threshold_transport=thresholds,
        explainability=dict(global_shap_additivity=shap["additivity"], shap_lime_agreement=agreements,
                            lime_fidelity_by_profile=fidelity.reset_index().to_dict("records"),
                            lime_limitation="Approximate local surrogate; substantial pointwise error and negative unbounded output in one profile despite stable feature sets. SHAP remains primary."),
        model_comparison_evidence=dict(baseline=table("baseline").to_dict("records"), nested=nested.to_dict("records")),
        temporal_data_reopened=False, predictive_experiments_rerun=False)


def boundary_checks(metadata):
    lo, hi = [metadata["reference_thresholds"][k] for k in ("sensitivity_90_reference", "sensitivity_80_reference")]
    return [dict(case=name, probability=float(p), expected=expected,
                 actual=model_derived_risk(p, metadata)["level"], passed=model_derived_risk(p, metadata)["level"] == expected)
            for name, p, expected in (("below_lower", np.nextafter(lo, -np.inf), "LOW"),
                                     ("exact_lower", lo, "MODERATE"), ("inside_moderate", (lo+hi)/2, "MODERATE"),
                                     ("exact_upper", hi, "HIGH"), ("above_upper", np.nextafter(hi, np.inf), "HIGH"))]


def language_findings(text):
    """Conservative lexical screen; this is not a proof of clinical safety."""
    findings, allowed = [], []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if re.search(r"\byou (?:do not |don't )?have diabetes\b", sentence, re.I):
            findings.append("direct diagnostic assertion/denial")
        if re.search(r"\bdiagnosed\b", sentence, re.I):
            remainder = re.sub(r"physician[-– ]diagnosed", "", sentence, flags=re.I)
            if re.search(r"\bdiagnosed\b", remainder, re.I):
                findings.append("unsupported diagnosed wording")
            else:
                allowed.append("physician-diagnosed target attribution")
        for match in re.finditer(r"clinically validated\b.{0,35}?\b(?:thresholds?|cutoffs?|categories|strata)\b", sentence, re.I):
            prefix = sentence[:match.start()]
            if re.search(r"\bnot\s+(?:diagnostic categories or\s+)?$", prefix, re.I):
                allowed.append("explicit denial of clinically validated cutoffs")
            else:
                findings.append("unsupported clinical-validation claim")
        for word in ("guaranteed", "safe", "critical", "emergency"):
            if re.search(r"\b" + word + r"\b", sentence, re.I):
                findings.append("prohibited term: " + word)
        if re.search(r"\b(?:metformin|insulin|semaglutide|glipizide)\b", sentence, re.I):
            findings.append("medication instruction/reference requires review")
        for match in re.finditer(r"\b(?:start|take|prescribe|increase|decrease)\b.{0,35}\b(?:medication|medicine|drug|dose)\b", sentence, re.I):
            prefix = sentence[:match.start()]
            direct_denial = re.search(r"\b(?:do not|does not|cannot|never)\s+$", prefix, re.I)
            coordinated_denial = re.search(r"\b(?:do not|does not)\b([^.!?;]*),\s+or\s+$", prefix, re.I)
            if direct_denial or (coordinated_denial and not re.search(r"\b(?:but|however|instead)\b", coordinated_denial.group(1), re.I)):
                allowed.append("explicit denial of medication prescribing")
            else:
                findings.append("medication instruction/reference requires review")
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|units?)\b(?!\s*/\s*d[lL])", sentence, re.I):
            findings.append("possible dosage instruction")
    return sorted(set(findings)), sorted(set(allowed))


def clinical_language_audit(results, config, smoke):
    texts = {}
    for rule in config["rules"]:
        for field in ("text", "trigger_description", "rationale"):
            texts[f"rule/{rule['rule_id']}/{field}"] = rule[field]
    for name, result in results.items():
        texts[f"backend/{name}/risk_disclaimer"] = result["model_derived_risk"]["disclaimer"]
        texts[f"backend/{name}/recommendation_disclaimer"] = result["recommendation_provenance"]["disclaimer"]
        for index, limitation in enumerate(result["limitations"]):
            texts[f"backend/{name}/limitation/{index}"] = limitation
    for index, text in enumerate(smoke["rendered_text"]):
        texts[f"streamlit/rendered/{index}"] = text
    rows = []
    for source, text in texts.items():
        findings, allowed = language_findings(text)
        rows.append(dict(audit_id=f"LANG_{len(rows)+1:04d}", source=source, text=text,
                         check="patient-facing prohibited-language screen", status="FLAGGED" if findings else "PASS",
                         matched_terms=json.dumps(findings), context_note="; ".join(allowed)))
    normal = "\n".join(smoke["rendered_text"])
    concepts = {
        "research_prototype": r"(?:research|academic).{0,30}prototype",
        "not_diagnosis": r"not a (?:medical )?diagnosis",
        "health_professional": r"does not replace a qualified healthcare professional",
        "not_clinical_cutoffs": r"not (?:diagnostic categories or )?clinically validated.{0,30}(?:thresholds|cutoffs)",
        "shap_not_causation": r"SHAP.{0,55}(?:not causation|not causal)",
    }
    for key, pattern in concepts.items():
        passed = bool(re.search(pattern, normal, re.I))
        rows.append(dict(audit_id="DISCLAIMER_" + key.upper(), source="streamlit/normal_output", text=key,
                         check="required visible disclaimer", status="PASS" if passed else "FLAGGED",
                         matched_terms="[]", context_note="Checked actual AppTest-rendered UI text"))
    return pd.DataFrame(rows)


TRACE_REQUIREMENTS = [
    ("OBJ1_PRED", "OBJ1", "Machine-learning diagnosis/risk prediction", "src/decision_support/nhanes_diabetes_cdss.py", "models/nhanes/phase2_xgboost_metadata.json", "PARTIAL", "Diagnosed-status classification implemented; not incident-risk prediction or clinical diagnosis."),
    ("OBJ1_COMPARE", "OBJ1", "Model comparison", "src/evaluation/nhanes_baseline_comparison.py", "results/nhanes/nested_tuning_summary.csv", "COMPLETE", "Saved LR/XGBoost baseline and nested-CV evidence; not rerun."),
    ("OBJ1_PERF", "OBJ1", "Performance evaluation", "src/evaluation/nhanes_temporal_validation.py", "results/nhanes/temporal_validation_metrics.csv", "COMPLETE", "Retrospective development/temporal evaluation, not prospective clinical validation."),
    ("OBJ2_FACTORS", "OBJ2", "Key factor identification", "src/explainability/nhanes_shap_global.py", "results/nhanes/xai/shap_global_core10.csv", "COMPLETE", "Model-associated features, not causal factors."),
    ("OBJ2_GLOBAL", "OBJ2", "Global explainability", "src/explainability/nhanes_shap_global.py", "results/nhanes/xai/figures/shap_global_core10_bar.png", "COMPLETE", "Saved development explanations; no recomputation."),
    ("OBJ2_LOCAL", "OBJ2", "Local explainability", "src/decision_support/nhanes_diabetes_cdss.py", "results/nhanes/cdss/final_end_to_end_scenarios.csv", "COMPLETE", "Fresh local SHAP inference/additivity tested."),
    ("OBJ2_SHAP", "OBJ2", "SHAP", "src/explainability/nhanes_shap_global.py", "results/nhanes/xai/shap_global_provenance.json", "COMPLETE", "Exact raw-margin additivity; signed original-feature aggregation."),
    ("OBJ2_LIME", "OBJ2", "LIME", "src/explainability/nhanes_lime_local.py", "results/nhanes/xai/lime_synthetic_explanations.csv", "PARTIAL", "Complementary implementation complete; limited fidelity precludes treating it as an exact account of predictions."),
    ("OBJ3_STRATA", "OBJ3", "Low/Moderate/High stratification", "src/decision_support/nhanes_recommendations.py", "results/nhanes/cdss/final_end_to_end_scenarios.csv", "COMPLETE", "All three project-defined model-derived strata demonstrated; no validated clinical cutoffs."),
    ("OBJ3_LIFESTYLE", "OBJ3", "Lifestyle guidance", "configs/nhanes_recommendation_rules.json", "results/nhanes/cdss/risk_recommendation_provenance.json", "COMPLETE", "Guideline-informed educational prompts only."),
    ("OBJ3_MONITOR", "OBJ3", "Monitoring guidance", "configs/nhanes_recommendation_rules.json", "results/nhanes/cdss/final_end_to_end_scenarios.csv", "COMPLETE", "Clinician-directed monitoring; no automatic schedule."),
    ("OBJ3_FOLLOWUP", "OBJ3", "Clinical follow-up guidance", "configs/nhanes_recommendation_rules.json", "results/nhanes/cdss/final_end_to_end_scenarios.csv", "COMPLETE", "Conditional review, not medical triage or diagnosis."),
    ("OBJ3_PATIENT", "OBJ3", "Patient-specific recommendations", "src/decision_support/nhanes_recommendations.py", "results/nhanes/cdss/final_end_to_end_scenarios.csv", "COMPLETE", "Observed input rules checked; SHAP does not drive actions."),
    ("OBJ4_INTEGRATION", "OBJ4", "Integrated CDSS", "app.py", "results/nhanes/cdss/final_streamlit_smoke.json", "COMPLETE", "Backend-to-presentation preservation verified."),
    ("OBJ4_UI", "OBJ4", "User-facing interface", "app.py", "results/nhanes/cdss/final_streamlit_smoke.json", "COMPLETE", "AppTest plus real localhost health check."),
    ("OBJ4_PRED", "OBJ4", "Prediction display", "app.py", "results/nhanes/cdss/final_streamlit_smoke.json", "COMPLETE", "Rendered percentage equals backend formatting."),
    ("OBJ4_EXPLAIN", "OBJ4", "Explanation display", "app.py", "results/nhanes/cdss/final_streamlit_smoke.json", "COMPLETE", "Core-10 SHAP figure and table render."),
    ("OBJ4_RISK", "OBJ4", "Risk-stratum display", "app.py", "results/nhanes/cdss/final_streamlit_smoke.json", "COMPLETE", "Mandatory model-derived labels/disclaimer render."),
    ("OBJ4_RECOMMEND", "OBJ4", "Recommendation display", "app.py", "results/nhanes/cdss/final_streamlit_smoke.json", "COMPLETE", "Four tabs and triggered-rule text render."),
    ("OBJ4_VALIDATE", "OBJ4", "Validation", "tests/test_nhanes_final_integrated.py", "results/nhanes/cdss/final_pytest_report.xml", "COMPLETE", "Software/prototype validation plus saved experimental evidence; clinical validation remains future work."),
    ("OBJ4_LIMITS", "OBJ4", "Clinical decision-support limitations", "app.py", "results/nhanes/cdss/clinical_language_audit.csv", "COMPLETE", "Actual rendered language screened; lexical audit is not clinical safety certification."),
]


def objective_traceability(runtime_ok=True):
    rows = []
    for req_id, objective, requirement, component, evidence, intended_status, note in TRACE_REQUIREMENTS:
        present = (ROOT/component).is_file() and (ROOT/evidence).is_file()
        status = intended_status if present and runtime_ok else "NOT IMPLEMENTED" if not present else "PARTIAL"
        rows.append(dict(requirement_id=req_id, synopsis_objective=objective, synopsis_requirement=requirement,
                         implemented_component=component, evidence_file=evidence,
                         validation_method="Repository evidence + integrated regression/scenario/UI audit", status=status, notes=note))
    return pd.DataFrame(rows)


def read_pytest_report(path):
    suites = ET.parse(path).getroot()
    cases = list(suites.iter("testcase"))
    return dict(test_count=len(cases), passed=sum(not any(c.tag in ("failure", "error", "skipped") for c in case) for case in cases),
                failures=len(list(suites.iter("failure"))), errors=len(list(suites.iter("error"))),
                skipped=len(list(suites.iter("skipped"))), report_file=str(path))


def build_validation_report(pytest_report, smoke_report, health_report, before, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("Final audit outputs use the fixed results/nhanes/cdss directory")
    test_results = read_pytest_report(pytest_report)
    smoke = json.loads(Path(smoke_report).read_text(encoding="utf-8"))
    assert test_results["test_count"] > 0 and not test_results["failures"] and not test_results["errors"]
    assert smoke["passed"] and health_report["passed"]
    service = NhanesDiabetesCDSS()
    integrity = verify_integrity(service)
    frame, full = validate_scenarios(service)
    audit = clinical_language_audit(full, service.recommendation_rules, smoke)
    boundaries = boundary_checks(service.metadata)
    summary = historical_summary()
    assert all(case["passed"] for case in boundaries)
    after = snapshot_inputs()
    assert before == after, "A frozen component or prior evidence file changed during validation"
    frame.to_csv(output_dir/"final_end_to_end_scenarios.csv", index=False)
    audit.to_csv(output_dir/"clinical_language_audit.csv", index=False)
    summary["current_integrated_validation"] = dict(
        scenario_count=len(frame), backend_repetitions_per_scenario=3, deterministic=True,
        model_integrity=integrity, boundary_checks=boundaries,
        clinical_language_flag_count=int((audit.status == "FLAGGED").sum()),
        pytest=test_results, streamlit_smoke=smoke["checks"], streamlit_server_health=health_report)
    (output_dir/"final_validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    trace = objective_traceability(runtime_ok=not (audit.status == "FLAGGED").any())
    trace.to_csv(output_dir/"objective_traceability.csv", index=False)
    provenance = dict(model_version=service.metadata["model_version"], artifact_hash=integrity["artifact_sha256"],
                      metadata_hash=integrity["metadata_sha256"], application_version="No release identifier; application source SHA-256 recorded",
                      application_sha256=sha256(ROOT/"app.py"), validation_timestamp=datetime.now(timezone.utc).isoformat(),
                      python_version=platform.python_version(), package_versions={p: version(p) for p in
                          ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "lime", "streamlit", "pytest")},
                      pytest=test_results, source_and_evidence_sha256_before=before, source_and_evidence_sha256_after=after,
                      validation_source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
                          (Path(__file__), ROOT/"scripts/validate_nhanes_cdss.py", ROOT/"tests/test_nhanes_final_integrated.py")},
                      files_used_for_validation=list(before)+[str(pytest_report), str(smoke_report)],
                      streamlit_server_health=health_report, raw_temporal_cohort_accessed=False,
                      predictive_experiments_rerun=False, frozen_model_retrained=False,
                      frozen_preprocessing_refit=False, thresholds_modified=False, recommendation_rules_modified=False,
                      patient_facing_output_changed=False, scenario_inputs=validation_scenarios(),
                      moderate_profile_method="One manually specified age-60 variant of the existing intermediate profile; no search or optimization",
                      traceability_status_counts=trace.status.value_counts().to_dict(),
                      clinical_language_flag_count=int((audit.status == "FLAGGED").sum()),
                      clinical_validation_claimed=False, clinical_language_audit_scope="Lexical/context checks and required disclaimer presence, not proof of clinical safety",
                      excluded_integration_test="tests/test_nhanes.py (raw temporal cohort access)")
    output_files = ["objective_traceability.csv", "final_end_to_end_scenarios.csv", "clinical_language_audit.csv", "final_validation_summary.json"]
    provenance["output_sha256"] = {name: sha256(output_dir/name) for name in output_files}
    (output_dir/"final_cdss_validation_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    if (audit.status == "FLAGGED").any():
        raise AssertionError("Patient-facing language needs review; flagged rows saved in clinical_language_audit.csv")
    return frame, trace, audit, summary, provenance

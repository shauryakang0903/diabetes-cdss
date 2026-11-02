"""Transparent post-prediction interpretation; no model fitting or cohort reads."""

import copy
import json
import math
from numbers import Real
from pathlib import Path

from src.data.preprocessing import CORE_FEATURES
from src.models.nhanes_phase2_model import MODEL_VERSION, load_metadata, sha256

FRAMEWORK_VERSION = "nhanes_recommendations_v1"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/nhanes_recommendation_rules.json"
SECTIONS = ("clinical_follow_up", "monitoring", "lifestyle", "feature_aware")
TRIGGERS = {"stratum", "always", "bmi_review", "current_smoker", "bp_review",
            "waist_review", "lipid_review", "any_missing", "scope_uncertain"}


def load_recommendation_rules():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["framework_version"] != FRAMEWORK_VERSION or config["model_version"] != MODEL_VERSION:
        raise ValueError("Recommendation configuration does not match this framework/model version")
    if config["risk_boundaries"] != {"lower_reference": "sensitivity_90_reference", "upper_reference": "sensitivity_80_reference"}:
        raise ValueError("Risk boundaries must name the frozen sensitivity references")
    ids = []
    for rule in config["rules"]:
        required = {"rule_id", "section", "trigger", "trigger_description", "text", "rationale", "source_labels"}
        if not required <= set(rule) or rule["section"] not in SECTIONS or rule["trigger"]["kind"] not in TRIGGERS:
            raise ValueError("Invalid recommendation rule definition")
        if not rule["source_labels"] or not set(rule["source_labels"]) <= set(config["sources"]):
            raise ValueError("Each recommendation requires known source labels")
        ids.append(rule["rule_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Recommendation rule IDs must be unique")
    config["config_sha256"] = sha256(CONFIG_PATH)
    return config


def model_derived_risk(probability, metadata=None, config=None):
    metadata = load_metadata() if metadata is None else metadata
    config = load_recommendation_rules() if config is None else config
    if metadata["model_version"] != config["model_version"]:
        raise ValueError("Model version does not match the recommendation framework")
    if isinstance(probability, bool) or not isinstance(probability, Real) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Classification probability must be finite and within [0,1]")
    names = config["risk_boundaries"]
    lower = metadata["reference_thresholds"][names["lower_reference"]]
    upper = metadata["reference_thresholds"][names["upper_reference"]]
    if not (0 < lower < upper < 1):
        raise ValueError("Frozen sensitivity references must satisfy 0 < reference90 < reference80 < 1")
    if probability < lower:
        level, lo, hi = "LOW", None, lower
    elif probability < upper:
        level, lo, hi = "MODERATE", lower, upper
    else:
        level, lo, hi = "HIGH", upper, None
    return dict(level=level, label=f"{level} model-derived risk", lower_boundary=lo, upper_boundary=hi,
                lower_inclusive=lo is not None, upper_inclusive=False if hi is not None else None,
                basis="Project-defined interpretation of frozen sensitivity-oriented development operating references; not incident diabetes risk.",
                boundary_source=config["risk_boundary_source"],
                references={names["lower_reference"]: lower, names["upper_reference"]: upper},
                disclaimer=config["disclaimer"])


def _triggered(trigger, level, inputs):
    """Small explicit predicate vocabulary; no eval or clinical inference."""
    def at_least(field, value):
        return inputs[field] is not None and inputs[field] >= value
    def below(field, value):
        return inputs[field] is not None and inputs[field] < value
    kind = trigger["kind"]
    adult = at_least("age", 20)
    sex_key = str(int(inputs["sex"])) if inputs["sex"] in (1, 2) else None
    if kind == "always":
        return True
    if kind == "stratum":
        return level in trigger["levels"]
    if kind == "any_missing":
        return any(value is None for value in inputs.values())
    if kind == "scope_uncertain":
        return not adult
    if kind == "current_smoker":
        return inputs["smoking_status"] == trigger["code"]
    if not adult:
        return False
    if kind == "bmi_review":
        return at_least("bmi", trigger["at_least"])
    if kind == "bp_review":
        return at_least("mean_systolic_bp", trigger["systolic_at_least"]) or at_least("mean_diastolic_bp", trigger["diastolic_at_least"])
    if kind == "waist_review":
        boundary = trigger["above_cm_by_sex"].get(sex_key)
        return boundary is not None and inputs["waist_cm"] is not None and inputs["waist_cm"] > boundary
    if kind == "lipid_review":
        boundary = trigger["hdl_below_by_sex"].get(sex_key)
        return at_least("total_cholesterol_mg_dl", trigger["total_at_least"]) or (boundary is not None and below("hdl_mg_dl", boundary))
    raise ValueError(f"Unknown recommendation trigger: {kind}")


def add_recommendation_layer(result, metadata=None, config=None):
    """Extend a backend result without mutating probability, SHAP, or inputs."""
    metadata = load_metadata() if metadata is None else metadata
    config = load_recommendation_rules() if config is None else config
    if result["model_version"] != metadata["model_version"]:
        raise ValueError("Prediction and risk-framework model versions differ")
    rows = result["all_feature_contributions"]
    if len(rows) != 10 or {r["feature"] for r in rows} != set(CORE_FEATURES):
        raise ValueError("Recommendation input must contain ten unique original Core-10 values")
    inputs = {row["feature"]: None if row["original_value"] is None else float(row["original_value"]) for row in rows}
    if any(value is not None and not math.isfinite(value) for value in inputs.values()):
        raise ValueError("Original measurements must be finite or missing")
    risk = model_derived_risk(result["predicted_probability"], metadata, config)
    recommendations = {section: [] for section in SECTIONS}
    for rule in config["rules"]:
        if _triggered(rule["trigger"], risk["level"], inputs):
            record = copy.deepcopy(rule)
            # URLs remain in developer configuration/docs, not patient output.
            record["source_labels"] = [config["sources"][key]["label"] for key in rule["source_labels"]]
            record["source_ids"] = list(rule["source_labels"])
            recommendations[rule["section"]].append(record)
    return dict(result, model_derived_risk=risk, recommendations=recommendations,
                recommendation_provenance=dict(
                    framework_version=config["framework_version"], date=config["date"],
                    config_sha256=config["config_sha256"], model_version=config["model_version"],
                    guideline_informed=True, clinician_review_required=True,
                    disclaimer=config["recommendation_disclaimer"],
                    triggered_rule_ids=[row["rule_id"] for section in SECTIONS for row in recommendations[section]],
                    input_policy=config["feature_rule_policy"]))

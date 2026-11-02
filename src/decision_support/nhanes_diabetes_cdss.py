"""Single-patient inference only: no cohort reads, fitting, or clinical cutoffs."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit

from src.data.preprocessing import CORE_FEATURES
from src.explainability.nhanes_shap_global import (
    aggregate_core10, create_tree_explainer, explain_transformed, feature_mapping,
)
from src.models.nhanes_phase2_model import (
    DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, load_frozen_model, load_metadata,
    predict_probabilities, validate_input_columns,
)
from src.decision_support.nhanes_recommendations import (
    add_recommendation_layer, load_recommendation_rules,
)
from src.decision_support.nhanes_input_quality import load_reference_metadata, assess_input_quality

DIRECTION_TOLERANCE = 1e-10
PROBABILITY_ATOL = 1e-6
PROBABILITY_RTOL = 1e-5
LIMITATIONS = (
    "This is model-based decision support, not a diagnosis of diabetes.",
    "Thresholds are development-derived operating references, not clinically validated clinical cutoffs.",
    "SHAP explains model behavior, not causation; contributions are log-odds, not probability changes.",
    "Incomplete inputs may reduce robustness: completed temporal validation showed lower performance among participants with missing Core-10 predictors.",
    "The target is physician-diagnosed diabetes (DIQ010), not prospective diabetes incidence; intended population is adults.",
)


def contribution_direction(value):
    if abs(value) <= DIRECTION_TOLERANCE:
        return "neutral"
    return "increases_model_score" if value > 0 else "decreases_model_score"


def reference_comparison(probability, thresholds):
    """Equality counts as meeting a reference; no new threshold selection."""
    comparisons = {name: {"threshold": float(value),
                         "at_or_above": bool(probability >= value)}
                   for name, value in thresholds.items()}
    above80 = comparisons["sensitivity_80_reference"]["at_or_above"]
    above90 = comparisons["sensitivity_90_reference"]["at_or_above"]
    if above80 and above90:
        text = "At or above both the 80%- and 90%-sensitivity development-derived screening references."
    elif above90:
        text = "At or above the 90%-sensitivity development-derived screening reference; below the 80%-sensitivity reference."
    elif above80:
        text = "At or above the 80%-sensitivity development-derived screening reference; below the 90%-sensitivity reference."
    else:
        text = "Below both sensitivity-oriented development-derived screening references."
    return comparisons, text


class NhanesDiabetesCDSS:
    """Load once per service instance; reuse for sequential patient requests.

    Inputs use harmonized NHANES codes. Missing values are passed to the frozen
    imputers. Unknown numeric category codes receive an applicability warning
    when the fitted encoder supports them. Prediction transformations are unchanged.
    """

    def __init__(self, artifact_path=DEFAULT_ARTIFACT_PATH, metadata_path=DEFAULT_METADATA_PATH):
        self.metadata = load_metadata(metadata_path)
        self.pipeline = load_frozen_model(artifact_path, metadata_path)
        self.preprocessor = self.pipeline.named_steps["preprocessing"]
        self.estimator = self.pipeline.named_steps["model"]
        self.mapping = feature_mapping(self.preprocessor)
        self.names = self.preprocessor.get_feature_names_out().tolist()
        categorical = self.preprocessor.named_transformers_["categorical"]
        columns = next(cols for name, _, cols in self.preprocessor.transformers_ if name == "categorical")
        self.categories = dict(zip(columns, categorical.named_steps["encoder"].categories_))
        self.explainer = create_tree_explainer(self.estimator, self.names)
        self.recommendation_rules = load_recommendation_rules()
        self.input_reference = load_reference_metadata(metadata_path=metadata_path, artifact_path=artifact_path)

    def predict(self, patient):
        """Return JSON-safe probability, operating references, and local SHAP."""
        if isinstance(patient, Mapping):
            if any(not pd.api.types.is_scalar(v) for k, v in patient.items() if k in CORE_FEATURES):
                raise ValueError("Patient values must be scalars or missing")
            patient = pd.DataFrame([patient])
        if not isinstance(patient, pd.DataFrame) or len(patient) != 1:
            raise ValueError("Expected a single patient mapping or a one-row DataFrame")
        if any(isinstance(v, (bool, np.bool_, complex)) for k, v in patient.iloc[0].items() if k in CORE_FEATURES):
            raise ValueError("Use numeric NHANES codes rather than boolean values")
        data = validate_input_columns(patient)
        for column, allowed in self.categories.items():
            value = data.iloc[0][column]
            if (pd.notna(value) and value not in allowed
                    and self.preprocessor.named_transformers_["categorical"].named_steps["encoder"].handle_unknown != "ignore"):
                raise ValueError(f"Invalid {column} code {value}; expected {allowed.tolist()} or missing")
        transformed = np.asarray(self.preprocessor.transform(data), dtype=float)
        if not np.isfinite(transformed).all():
            raise ValueError("Frozen preprocessing produced nonfinite values")
        probability = float(predict_probabilities(data, pipeline=self.pipeline)[0])
        values, base, margin, diagnostics, _ = explain_transformed(
            self.estimator, transformed, self.names, explainer=self.explainer)
        core = aggregate_core10(values, self.mapping)[0]
        reconstructed_margin = float(base[0] + core.sum())
        reconstructed_probability = float(expit(reconstructed_margin))
        np.testing.assert_allclose(reconstructed_probability, probability,
                                   atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL)
        rows = []
        for feature, value in zip(CORE_FEATURES, core):
            original = patient.iloc[0][feature]
            if pd.isna(original):
                original = None
            elif not isinstance(original, str):
                original = float(data.iloc[0][feature])
            rows.append(dict(feature=feature, original_value=original,
                             was_missing=bool(pd.isna(data.iloc[0][feature])),
                             shap_contribution=float(value), absolute_shap_magnitude=float(abs(value)),
                             direction=contribution_direction(value)))
        # Python's stable sort preserves Core-10 order for exact ties.
        rows.sort(key=lambda row: -row["absolute_shap_magnitude"])
        thresholds, interpretation = reference_comparison(probability, self.metadata["reference_thresholds"])
        result = dict(
            model_version=self.metadata["model_version"], predicted_probability=probability,
            predicted_probability_percent=100 * probability,
            default_classification=int(probability >= 0.5),
            default_classification_context="Technical reference at 0.5 only; not a diagnosis.",
            reference_interpretation=interpretation, thresholds=thresholds,
            top_increasing_contributors=[r for r in rows if r["direction"] == "increases_model_score"][:3],
            top_decreasing_contributors=[r for r in rows if r["direction"] == "decreases_model_score"][:3],
            all_feature_contributions=rows,
            explanation_integrity=dict(
                scale="raw_margin_log_odds", base_margin=float(base[0]), raw_margin=float(margin[0]),
                reconstructed_margin=reconstructed_margin, transformed_shap_total=float(values.sum()),
                core10_shap_total=float(core.sum()), transformed_feature_count=18, original_feature_count=10,
                margin_additivity=diagnostics, reconstructed_probability=reconstructed_probability,
                probability_absolute_error=abs(reconstructed_probability-probability),
                probability_absolute_tolerance=PROBABILITY_ATOL, probability_relative_tolerance=PROBABILITY_RTOL,
                probability_check_passed=True, direction_tolerance=DIRECTION_TOLERANCE),
            limitations=list(LIMITATIONS))
        result = add_recommendation_layer(result, self.metadata, self.recommendation_rules)
        result['input_quality'] = assess_input_quality(data, self.input_reference)
        return result

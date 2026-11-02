"""Development-reference LIME for the frozen predictor, never model development."""

from itertools import combinations

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.data.nhanes_model_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data.preprocessing import CORE_FEATURES
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.evaluation.nhanes_preprocessing_comparison import load_development_data
from src.models.nhanes_phase2_model import predict_probabilities, validate_input_columns

SEED = 42
NUM_SAMPLES = 5000
STABILITY_SEEDS = (42, 43, 44, 45, 46)
NEUTRAL_TOLERANCE = 1e-10
KERNEL_WIDTH = 0.75 * np.sqrt(len(CORE_FEATURES))


def weight_direction(weight):
    if abs(weight) <= NEUTRAL_TOLERANCE:
        return "neutral"
    return "increases_model_probability" if weight > 0 else "decreases_model_probability"


def terms_to_core10(explanation, original_values, imputed_values):
    """Map LIME's integer feature IDs, never parse rounded interval strings."""
    terms = explanation.local_exp[1]
    if sorted(index for index, _ in terms) != list(range(len(CORE_FEATURES))):
        raise ValueError("LIME must explain every original Core-10 feature exactly once")
    mapped = explanation.domain_mapper.map_exp_ids(terms)
    rows = []
    for (index, weight), (term, _) in zip(terms, mapped):
        if not np.isfinite(weight):
            raise ValueError("Nonfinite LIME coefficient")
        feature = CORE_FEATURES[index]
        rows.append(dict(feature=feature, original_value=original_values[feature],
                         was_missing=original_values[feature] is None,
                         reference_imputed_value=float(imputed_values[index]), lime_term=term,
                         lime_weight=float(weight), absolute_local_importance=float(abs(weight)),
                         direction=weight_direction(weight)))
    rows.sort(key=lambda r: (-r["absolute_local_importance"], CORE_FEATURES.index(r["feature"])))
    return [dict(rank=rank, **row) for rank, row in enumerate(rows, 1)]


class NhanesLimeLocal:
    """Use the entire fixed development cohort for LIME reference statistics.

    Fitted imputers transform reference values without fitting. Noncontiguous
    NHANES category codes are represented by contiguous LIME indices, then
    decoded before every call to the existing frozen probability helper.
    A fresh seeded LIME explainer per call prevents RNG call-order dependence.
    """

    def __init__(self, backend=None):
        self.backend = NhanesDiabetesCDSS() if backend is None else backend
        self.preprocessor = self.backend.preprocessor
        self.category_codes = {feature: np.asarray(codes, dtype=float).copy()
                               for feature, codes in self.backend.categories.items()}
        self.categorical_indices = [CORE_FEATURES.index(c) for c in CATEGORICAL_FEATURES]
        development = load_development_data()  # Fixed path; no alternative cohort path accepted.
        self.reference_n = len(development)
        self.reference_data = self._encode(self._impute(validate_input_columns(development)))

    def _impute(self, frame):
        """Frozen imputation only, retaining the original ten-feature space."""
        result = frame.copy()
        numeric = self.preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        categorical = self.preprocessor.named_transformers_["categorical"].named_steps["imputer"]
        result[NUMERIC_FEATURES] = numeric.transform(frame[NUMERIC_FEATURES])
        result[CATEGORICAL_FEATURES] = categorical.transform(frame[CATEGORICAL_FEATURES])
        return result[CORE_FEATURES]

    def _encode(self, frame):
        array = frame[CORE_FEATURES].to_numpy(dtype=float, copy=True)
        for feature, codes in self.category_codes.items():
            index = CORE_FEATURES.index(feature)
            if not np.isin(array[:, index], codes).all():
                raise ValueError(f"Unexpected {feature} code in LIME input/reference data")
            lookup = {code: position for position, code in enumerate(codes)}
            array[:, index] = [lookup[value] for value in array[:, index]]
        if not np.isfinite(array).all():
            raise ValueError("LIME reference data must be finite after frozen imputation")
        return array

    def _decode(self, samples):
        array = np.asarray(samples, dtype=float).copy()
        if array.ndim != 2 or array.shape[1] != len(CORE_FEATURES) or not np.isfinite(array).all():
            raise ValueError("Expected finite LIME samples with ten original features")
        for feature, codes in self.category_codes.items():
            index = CORE_FEATURES.index(feature)
            encoded = array[:, index]
            if not ((encoded == np.floor(encoded)) & (encoded >= 0) & (encoded < len(codes))).all():
                raise ValueError(f"Invalid LIME categorical index for {feature}")
            array[:, index] = codes[encoded.astype(int)]
        return pd.DataFrame(array, columns=CORE_FEATURES)

    def predict_lime_samples(self, samples):
        """LIME callback delegates to the existing fitted pipeline helper."""
        probabilities = predict_probabilities(self._decode(samples), pipeline=self.backend.pipeline)
        return np.column_stack((1 - probabilities, probabilities))

    def build_explainer(self, random_seed=SEED):
        return LimeTabularExplainer(
            self.reference_data, mode="classification", feature_names=list(CORE_FEATURES),
            class_names=["No physician-diagnosed diabetes", "Physician-diagnosed diabetes"],
            categorical_features=self.categorical_indices,
            categorical_names={CORE_FEATURES.index(c): [f"{value:g}" for value in codes]
                               for c, codes in self.category_codes.items()},
            discretize_continuous=True, discretizer="quartile", feature_selection="none",
            kernel_width=KERNEL_WIDTH, random_state=int(random_seed),
            sample_around_instance=False)

    def explain_patient(self, patient, *, random_seed=SEED, num_samples=NUM_SAMPLES):
        if not isinstance(num_samples, int) or num_samples < 100:
            raise ValueError("LIME requires an integer num_samples >= 100")
        # Reuse authoritative single-patient validation and the existing SHAP output.
        shap_result = self.backend.predict(patient)
        original = {r["feature"]: r["original_value"] for r in shap_result["all_feature_contributions"]}
        frame = pd.DataFrame([original], columns=CORE_FEATURES)
        imputed = self._impute(validate_input_columns(frame))
        row = self._encode(imputed)[0]
        explainer = self.build_explainer(random_seed)
        explanation = explainer.explain_instance(
            row, self.predict_lime_samples, labels=(1,), num_features=10,
            num_samples=num_samples, distance_metric="euclidean")
        probability = shap_result["predicted_probability"]
        np.testing.assert_allclose(explanation.predict_proba[1], probability, atol=1e-7, rtol=0)
        weights = terms_to_core10(explanation, original, imputed.iloc[0].to_numpy(dtype=float))
        intercept = float(explanation.intercept[1])
        surrogate_prediction = float(np.asarray(explanation.local_pred).reshape(-1)[0])
        np.testing.assert_allclose(intercept + sum(r["lime_weight"] for r in weights),
                                   surrogate_prediction, atol=1e-10, rtol=1e-10)
        if not np.isfinite([intercept, surrogate_prediction, explanation.score]).all():
            raise ValueError("Nonfinite LIME surrogate diagnostics")
        return dict(model_version=shap_result["model_version"], predicted_probability=probability,
                    random_seed=int(random_seed), num_samples=num_samples, local_intercept=intercept,
                    surrogate_prediction=surrogate_prediction, local_weighted_r2=float(explanation.score),
                    surrogate_absolute_error=abs(surrogate_prediction-probability),
                    local_feature_weights=weights, shap_result=shap_result,
                    limitations=["LIME fits a local weighted surrogate to frozen-model probability outputs, not patient outcomes.",
                                 "Coefficients describe active feature conditions, not clinical effects or probability changes from an intervention.",
                                 "Surrogate predictions may differ from the frozen model and are not clipped to [0,1].",
                                 "Perturbations can violate correlations and yield implausible profiles; explanations are not clinically validated."])


def compare_shap_lime(shap_result, lime_result):
    """Compare absolute-importance ranks; directions refer to different baselines/scales."""
    shap = shap_result["all_feature_contributions"]
    lime = lime_result["local_feature_weights"]
    for rows in (shap, lime):
        if len(rows) != 10 or {r["feature"] for r in rows} != set(CORE_FEATURES):
            raise ValueError("Comparison requires ten unique original Core-10 features")
    shap = sorted(shap, key=lambda r: (-abs(r["shap_contribution"]), CORE_FEATURES.index(r["feature"])))
    lime = sorted(lime, key=lambda r: (-abs(r["lime_weight"]), CORE_FEATURES.index(r["feature"])))
    s = {r["feature"]: r["shap_contribution"] for r in shap}
    l = {r["feature"]: r["lime_weight"] for r in lime}
    def same_direction(feature):
        return weight_direction(s[feature]) == weight_direction(l[feature])
    result = {}
    for k in (3, 5):
        s_top, l_top = [r["feature"] for r in shap[:k]], [r["feature"] for r in lime[:k]]
        common = [c for c in CORE_FEATURES if c in set(s_top) & set(l_top)]
        result.update({f"shap_top{k}": s_top, f"lime_top{k}": l_top,
                       f"top{k}_overlap_count": len(common), f"top{k}_overlap": len(common)/k,
                       f"top{k}_overlapping_features": common,
                       f"top{k}_direction_agreement": float(np.mean([same_direction(c) for c in common])) if common else None})
    result["direction_agreement_all_features"] = float(np.mean([same_direction(c) for c in CORE_FEATURES]))
    return result


def summarize_stability(explanations):
    """Selection frequencies and mean pairwise top-k overlaps across fixed seeds."""
    if len(explanations) < 2 or len({e["random_seed"] for e in explanations}) != len(explanations):
        raise ValueError("Stability requires at least two distinct fixed seeds")
    sets = {k: [{r["feature"] for r in e["local_feature_weights"][:k]} for e in explanations] for k in (3, 5)}
    pairs = list(combinations(range(len(explanations)), 2))
    rows = []
    for feature in CORE_FEATURES:
        row = dict(feature=feature, seed_count=len(explanations))
        for k in (3, 5):
            count = sum(feature in selected for selected in sets[k])
            row.update({f"top{k}_selection_count": count, f"top{k}_selection_frequency": count/len(explanations),
                        f"mean_pairwise_top{k}_overlap": float(np.mean([len(sets[k][a] & sets[k][b])/k for a, b in pairs]))})
        rows.append(row)
    return rows

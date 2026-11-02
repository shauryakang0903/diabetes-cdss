"""Presentation only: predictions, thresholds and SHAP come from the backend."""

import pandas as pd

from src.explainability.nhanes_shap_global import LABELS

CATEGORY_LABELS = {
    "sex": {1: "Male", 2: "Female"},
    "race_ethnicity": {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                       4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other race / multiracial"},
    "smoking_status": {0: "Never smoker", 1: "Former smoker", 2: "Current smoker"},
}
UNITS = {"age": "years", "bmi": "kg/m²", "waist_cm": "cm",
         "mean_systolic_bp": "mmHg", "mean_diastolic_bp": "mmHg",
         "hdl_mg_dl": "mg/dL", "total_cholesterol_mg_dl": "mg/dL"}
REFERENCE_LABELS = {
    "default": "Default 0.50 technical reference",
    "sensitivity_80_reference": "~80%-sensitivity development reference",
    "sensitivity_90_reference": "~90%-sensitivity development reference",
    "youden_reference": "Youden statistical reference",
}


def assess_patient(patient, service):
    """Single integration boundary; never calculate predictions in presentation."""
    return service.predict(patient)


def probability_text(result):
    return f"{result['predicted_probability_percent']:.2f}%"


def display_value(row):
    value, feature = row["original_value"], row["feature"]
    if value is None:
        return "Unavailable (imputed by frozen pipeline)"
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature].get(value, str(value))
    return f"{float(value):g} {UNITS.get(feature, '')}".strip()


def explanation_table(result):
    return pd.DataFrame([
        {"Core-10 feature": row["feature"], "Feature": LABELS[row["feature"]],
         "Entered value": display_value(row), "SHAP contribution (log-odds)": row["shap_contribution"],
         "Direction": row["direction"].replace("_", " ")}
        for row in result["all_feature_contributions"]])


def reference_table(result):
    return pd.DataFrame([
        {"Operating reference": REFERENCE_LABELS[name], "Threshold": row["threshold"],
         "Input profile": "At or above" if row["at_or_above"] else "Below"}
        for name, row in result["thresholds"].items()])


def local_explanation_figure(result):
    """Plot signed, already-aggregated Core-10 contributions without recalculation."""
    from matplotlib.figure import Figure
    rows = list(reversed(result["all_feature_contributions"]))
    values = [r["shap_contribution"] for r in rows]
    colors = [{"increases_model_score": "#b34d34", "decreases_model_score": "#287da0",
               "neutral": "#7b828a"}[r["direction"]] for r in rows]
    fig = Figure(figsize=(9, 5.4), layout="constrained")
    ax = fig.subplots()
    ax.barh([LABELS[r["feature"]] for r in rows], values, color=colors)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Signed SHAP contribution to model margin / log-odds")
    ax.set_title("Local explanation for the submitted input profile", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    return fig

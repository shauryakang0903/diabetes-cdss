"""Launch from the repository root: python -m streamlit run app.py."""

import logging
from pathlib import Path
from threading import RLock

import streamlit as st

from scripts.demo_nhanes_cdss import synthetic_profiles
from src.data.preprocessing import CORE_FEATURES
from src.decision_support.nhanes_diabetes_cdss import NhanesDiabetesCDSS
from src.decision_support.nhanes_recommendations import FRAMEWORK_VERSION
from src.decision_support.nhanes_input_quality import REFERENCE_VERSION
from src.decision_support.prototype_presentation import (
    CATEGORY_LABELS, UNITS, LABELS, assess_patient, display_value, explanation_table,
    local_explanation_figure, probability_text, reference_table,
)

ROOT = Path(__file__).resolve().parent
PROFILES = synthetic_profiles()
PROFILE_LABELS = {
    "relatively_lower_scoring_example": "Synthetic example: relatively lower scoring",
    "intermediate_scoring_example": "Synthetic example: intermediate scoring",
    "higher_scoring_example": "Synthetic example: higher scoring",
}


@st.cache_resource(show_spinner=False)
def get_backend(framework_version=FRAMEWORK_VERSION, input_reference_version=REFERENCE_VERSION):
    # Shared resource contains no patient records. Serialize access to the cached
    # SHAP explainer, whose shared threaded use is not promised by the backend.
    service = NhanesDiabetesCDSS()
    if service.recommendation_rules["framework_version"] != framework_version:
        raise ValueError("Cached backend recommendation framework version differs")
    if service.input_reference['reference_version'] != input_reference_version:
        raise ValueError('Cached backend input reference version differs')
    return service, RLock()


def load_profile():
    for feature, value in PROFILES[st.session_state["demo_profile"]].items():
        st.session_state[feature] = float(value) if feature in UNITS else value
    st.session_state.pop("assessment", None)
    st.session_state['unavailable_measurements'] = []


def clear_assessment():
    st.session_state.pop('assessment', None)


def show_contributors(title, rows):
    st.markdown(f"#### {title}")
    if not rows:
        st.caption("No contributors in this direction for this profile.")
    for row in rows:
        st.markdown(f"**{LABELS[row['feature']]}:** {display_value(row)}")
        st.caption(f"Signed SHAP contribution: {row['shap_contribution']:+.4f} log-odds")


def show_result(result):
    st.subheader("Prediction result")
    st.caption("Results describe the last submitted inputs. Submit again after changing measurements.")
    st.metric("Model-estimated probability", probability_text(result))
    st.write("The frozen model estimates a diabetes-classification probability of "
             f"{probability_text(result)} for this input profile.")
    st.caption(f"Model version: {result['model_version']}")
    quality = result['input_quality']
    st.subheader('Input Data Quality')
    st.write(f"Input completeness: {quality['observed_features']}/10 ({quality['completeness_percent']:.0f}%) · "
             f"{quality['completeness_label']}")
    st.write(quality['overall_applicability'])
    st.info(quality['applicability_note'])
    for key, label in [('missing_feature_names', 'Unavailable'), ('uncommon_features', 'Uncommon'),
                       ('out_of_range_features', 'Outside development range'), ('unseen_categories', 'Unseen categories')]:
        if quality[key]:
            st.write(f"{label}: " + ', '.join(LABELS[f] for f in quality[key]))
    st.caption(quality['disclaimer'])
    with st.expander('Detailed input development-range checks'):
        for feature, detail in quality['feature_status'].items():
            st.write(f"{LABELS[feature]}: {detail['status']} · entered value: {detail['value']}")
        st.caption('Project-defined interface labels based on raw observed development inputs; no clinical confidence estimate.')
    risk = result["model_derived_risk"]
    st.subheader("Model-derived risk stratum")
    st.markdown(f"### {risk['label']}")
    st.write("Model-derived decision-support category — not a diagnosis.")
    st.caption(risk["disclaimer"])
    st.subheader("Development-reference interpretation")
    st.info(result["reference_interpretation"])
    st.dataframe(reference_table(result), hide_index=True, width="stretch")
    st.caption("These are development-derived model operating references and are not clinically validated diagnostic thresholds.")
    st.subheader("Why the model predicted this — SHAP")
    st.write("These features increased or decreased the model score relative to its SHAP base for this input. "
             "They do not establish causes of diabetes or changes in probability.")
    left, right = st.columns(2)
    with left:
        show_contributors("Factors increasing the model score", result["top_increasing_contributors"])
    with right:
        show_contributors("Factors decreasing the model score", result["top_decreasing_contributors"])
    fig = local_explanation_figure(result)
    st.pyplot(fig, width="stretch")
    st.caption("Orange: increases model score; blue: decreases model score; gray: neutral. Values are log-odds contributions, not probability changes.")
    with st.expander("Complete Core-10 explanation and integrity checks"):
        st.dataframe(explanation_table(result), hide_index=True, width="stretch")
        integrity = result["explanation_integrity"]
        st.write("Explanation additivity and reconstructed-probability checks passed.")
        st.caption(f"Margin reconstruction error: {integrity['margin_additivity']['maximum_absolute_error']:.3g}; "
                   f"probability reconstruction error: {integrity['probability_absolute_error']:.3g}.")
    st.subheader("Suggested decision-support actions")
    st.caption("These guideline-informed prompts use the stratum and entered measurements. "
               "SHAP explains the model; its directions do not determine clinical actions.")
    sections = (("clinical_follow_up", "Clinical follow-up"), ("monitoring", "Monitoring"),
                ("lifestyle", "Lifestyle"), ("feature_aware", "Patient-specific considerations"))
    for tab, (key, _) in zip(st.tabs([label for _, label in sections]), sections):
        with tab:
            rows = result["recommendations"][key]
            if not rows:
                st.write("No additional measurement-specific rule was triggered. This does not establish normal health or remove the need for clinical review.")
            for row in rows:
                st.markdown(f"- {row['text']}")
                st.caption(f"{row['rule_id']} · {'; '.join(row['source_labels'])}")
    st.caption(result["recommendation_provenance"]["disclaimer"])
    with st.expander("Recommendation audit details"):
        st.caption(f"Framework: {result['recommendation_provenance']['framework_version']} · "
                   f"Reviewed: {result['recommendation_provenance']['date']}")
        for key, _ in sections:
            for row in result["recommendations"][key]:
                st.write(f"{row['rule_id']}: {row['trigger_description']}")
                st.caption(row["rationale"])
    for limitation in result["limitations"]:
        st.caption(limitation)


def main():
    st.set_page_config(page_title="Explainable Diabetes CDSS", page_icon="🔬", layout="wide")
    st.title("Explainable AI Clinical Decision Support System for Diabetes Risk Assessment")
    st.caption("NHANES-based · Frozen XGBoost · Local SHAP · Evaluated in a separate temporal cohort")
    st.warning("Research / academic prototype. Model output is decision support, not a medical diagnosis, "
               "and does not replace a qualified healthcare professional.")
    with st.expander("About the system"):
        st.write("This prototype demonstrates a frozen model for physician-diagnosed diabetes classification "
                 "using ten adult health measurements and characteristics. It does not estimate prospective "
                 "diabetes incidence. A separate rule-based layer adds model-derived strata and guideline-informed "
                 "follow-up, monitoring and lifestyle prompts; it does not diagnose or prescribe treatment.")
    st.subheader("Patient information input")
    st.caption("Synthetic demonstration profiles are software examples, not clinically validated patient archetypes. "
               "Age, sex and race/ethnicity are required. Mark unavailable clinical measurements below; "
               "the frozen imputers handle missing values. No patient identifiers are requested.")
    st.selectbox("Synthetic demonstration profiles", list(PROFILES), index=1,
                 format_func=PROFILE_LABELS.get, key="demo_profile", on_change=load_profile)
    if "age" not in st.session_state:
        load_profile()
    unavailable = st.multiselect('Unavailable clinical measurements',
                                [f for f in CORE_FEATURES if f not in ('age', 'sex', 'race_ethnicity')],
                                format_func=LABELS.get, key='unavailable_measurements', on_change=clear_assessment,
                                help='Choose availability before editing the form. Selected values are submitted as missing.')
    with st.form("patient_assessment"):
        columns = st.columns(2)
        patient = {}
        for index, feature in enumerate(CORE_FEATURES):
            with columns[index % 2]:
                if feature in CATEGORY_LABELS:
                    patient[feature] = st.selectbox(LABELS[feature], list(CATEGORY_LABELS[feature]),
                                                  format_func=CATEGORY_LABELS[feature].get, key=feature, disabled=feature in unavailable)
                else:
                    patient[feature] = st.number_input(f"{LABELS[feature]} ({UNITS[feature]})", key=feature,
                                                       step=1.0 if feature == "age" else 0.1, format="%.1f", disabled=feature in unavailable)
                if feature in unavailable:
                    patient[feature] = None
        submitted = st.form_submit_button("Run Diabetes Assessment", type="primary")
    if submitted:
        st.session_state.pop("assessment", None)
        try:
            with st.spinner("Loading verified frozen model…"):
                service, lock = get_backend()
        except Exception:
            logging.exception("Frozen CDSS artifact could not be loaded")
            st.error("The frozen model or its dependencies could not be loaded or verified. "
                     "Check the artifact, metadata and documented environment; then restart the app.")
        else:
            try:
                with st.spinner("Calculating prediction and local explanation…"), lock:
                    st.session_state["assessment"] = assess_patient(patient, service)
            except (ValueError, TypeError) as exc:
                st.error(f"Please check the submitted input fields: {exc}")
            except Exception:
                logging.exception("CDSS prediction or explanation failed")
                st.error("The prediction or explanation integrity check could not be completed. "
                         "No assessment is displayed. Ask the prototype maintainer to check the application log.")
    if "assessment" in st.session_state:
        show_result(st.session_state["assessment"])
    with st.expander("How the model works globally"):
        st.write("Global importance describes overall model behavior across development participants. "
                 "An individual explanation can differ. These are previously saved results; no global SHAP computation runs here.")
        image = ROOT / "results/nhanes/xai/figures/shap_global_core10_bar.png"
        if image.is_file():
            st.image(str(image), caption="Signed one-hot contributions were aggregated to Core-10 before calculating global mean absolute SHAP.")
        else:
            st.info("The saved global explanation figure is unavailable in this installation.")
    with st.expander("Model and validation information"):
        st.write("Development: NHANES 2011–March 2020, N = 25,580 adults. "
                 "Separate temporal evaluation: NHANES 2021–2023, N = 7,536 adults.")
        st.write("Previously recorded frozen XGBoost temporal results: ROC-AUC ≈ 0.812; "
                 "average precision ≈ 0.425; Brier score ≈ 0.100.")
        st.caption("These summary results do not establish clinical utility or regulatory approval. "
                   "Temporal performance was weaker among participants with missing predictor measurements. "
                   "The application does not load either cohort.")
    st.subheader("Important limitations")
    st.write("Development-derived references are not clinical diagnostic cutoffs. SHAP describes model behavior, "
             "not causation. Performance may be lower with missing measurements. A clinician may require "
             "laboratory testing and information outside this model. Suggested actions are educational prompts, "
             "not an individualized treatment plan; no medication is prescribed.")
    st.caption("This demonstration does not save submitted records to disk. Session results are held in memory. "
               "Use synthetic inputs during academic demonstrations; this is not a production clinical deployment.")


if __name__ == "__main__":
    main()

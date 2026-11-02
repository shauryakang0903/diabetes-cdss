"""Global exact Tree SHAP for the existing frozen artifact; no fitting anywhere."""

import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import DMatrix

from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_preprocessing_comparison import DEVELOPMENT_PATH, OUTPUT_DIR as RESULTS_DIR, load_development_data
from src.models.nhanes_phase2_model import (
    DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, load_frozen_model, load_metadata,
    sha256, validate_input_columns,
)

XAI_DIR = RESULTS_DIR / "xai"
SEED = 42
ADDITIVITY_ATOL = 1e-4
ADDITIVITY_RTOL = 1e-5
LABELS = {
    "age": "Age", "sex": "Sex", "race_ethnicity": "Race/ethnicity", "bmi": "BMI",
    "waist_cm": "Waist circumference", "mean_systolic_bp": "Mean systolic BP",
    "mean_diastolic_bp": "Mean diastolic BP", "hdl_mg_dl": "HDL cholesterol",
    "total_cholesterol_mg_dl": "Total cholesterol", "smoking_status": "Smoking status",
}


def feature_mapping(preprocessor):
    """Use fitted transformer slices/categories, never split underscore-rich names."""
    names = preprocessor.get_feature_names_out().tolist()
    if len(names) != 18:
        raise ValueError(f"Frozen model must have 18 transformed features, found {len(names)}")
    transformers = {name: (transformer, list(columns)) for name, transformer, columns
                    in preprocessor.transformers_ if name in ("numeric", "categorical")}
    numeric, numeric_columns = transformers["numeric"]
    categorical, categorical_columns = transformers["categorical"]
    numeric_slice, cat_slice = preprocessor.output_indices_["numeric"], preprocessor.output_indices_["categorical"]
    if numeric_slice.stop - numeric_slice.start != len(numeric_columns):
        raise ValueError("Numeric outputs no longer map one-to-one to frozen inputs")
    encoder = categorical.named_steps["encoder"]
    if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
        raise ValueError("Expected frozen one-hot encoding without dropped/grouped categories")
    rows = []
    for i, original in zip(range(numeric_slice.start, numeric_slice.stop), numeric_columns):
        rows.append(dict(transformed_feature=names[i], transformed_index=i,
                         original_core10_feature=original, category_value=np.nan,
                         display_label=LABELS[original]))
    index = cat_slice.start
    encoded_names = encoder.get_feature_names_out(categorical_columns)
    for original, categories in zip(categorical_columns, encoder.categories_):
        for category in categories:
            if names[index] != "categorical__" + encoded_names[index-cat_slice.start]:
                raise ValueError("Frozen encoded names do not match fitted category order")
            rows.append(dict(transformed_feature=names[index], transformed_index=index,
                             original_core10_feature=original, category_value=float(category),
                             display_label=f"{LABELS[original]} = {category:g}"))
            index += 1
    mapping = pd.DataFrame(rows).sort_values("transformed_index").reset_index(drop=True)
    if index != cat_slice.stop or mapping.transformed_index.tolist() != list(range(18)) or (
            set(mapping.original_core10_feature) != set(CORE_FEATURES)):
        raise ValueError("Mapping must cover all 18 transformed columns and every Core-10 feature")
    return mapping


def aggregate_core10(values, mapping):
    """Sum signed one-hot contributions per participant BEFORE taking absolutes."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(mapping) or not np.isfinite(values).all():
        raise ValueError("SHAP values must be a finite participant-by-transformed-feature matrix")
    if sorted(mapping.transformed_index.tolist()) != list(range(values.shape[1])) or (
            set(mapping.original_core10_feature) != set(CORE_FEATURES)):
        raise ValueError("Feature mapping must cover all transformed columns exactly once")
    aggregated = np.column_stack([
        values[:, mapping.loc[mapping.original_core10_feature == feature, "transformed_index"].to_numpy()].sum(axis=1)
        for feature in CORE_FEATURES])
    np.testing.assert_allclose(aggregated.sum(axis=1), values.sum(axis=1), rtol=1e-10, atol=1e-10)
    return aggregated


def global_summary(values, feature_names, feature_column):
    """Population SD (ddof=0) over explained participants; deterministic tie ranks."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names) or not np.isfinite(values).all():
        raise ValueError("Invalid SHAP dimensions or nonfinite SHAP values")
    result = pd.DataFrame({feature_column: feature_names, "feature_index": np.arange(len(feature_names)),
        "mean_absolute_shap": np.abs(values).mean(axis=0), "mean_signed_shap": values.mean(axis=0),
        "sd_shap": values.std(axis=0, ddof=0)})
    result = result.sort_values(["mean_absolute_shap", "feature_index"], ascending=[False, True], kind="stable")
    result.insert(0, "rank", np.arange(1, len(result)+1))
    return result.reset_index(drop=True)


def verify_additivity(values, base, margin):
    values, base, margin = np.asarray(values, dtype=float), np.asarray(base, dtype=float), np.asarray(margin, dtype=float)
    if values.ndim != 2 or base.shape != (len(values),) or margin.shape != (len(values),) or not all(
            np.isfinite(array).all() for array in (values, base, margin)):
        raise ValueError("Invalid shapes/nonfinite values for SHAP additivity verification")
    reconstructed = base + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, margin, rtol=ADDITIVITY_RTOL, atol=ADDITIVITY_ATOL)
    error = np.abs(reconstructed-margin)
    return dict(maximum_absolute_error=float(error.max()), mean_absolute_error=float(error.mean()),
                absolute_tolerance=ADDITIVITY_ATOL, relative_tolerance=ADDITIVITY_RTOL, passed=True)


def create_tree_explainer(estimator, names):
    """Construct the shared frozen-model explainer, reusable across patient calls."""
    return shap.TreeExplainer(estimator.get_booster(), model_output="raw",
                             feature_perturbation="tree_path_dependent", feature_names=names)


def explain_transformed(estimator, transformed, names, *, explainer=None):
    """Exact tree-path-dependent SHAP with no external/background dataset."""
    if explainer is None:
        explainer = create_tree_explainer(estimator, names)
    explanation = explainer(transformed, check_additivity=True, approximate=False)
    values = np.asarray(explanation.values, dtype=float)
    if values.shape != transformed.shape or not np.isfinite(values).all():
        raise ValueError("Unexpected binary raw-margin SHAP output shape or values")
    expected = np.asarray(explainer.expected_value, dtype=float)
    if expected.size != 1:
        raise ValueError("Expected one base value for the binary XGBoost margin")
    base = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if base.size == 1:
        base = np.full(len(values), float(expected.reshape(-1)[0]))
    np.testing.assert_allclose(base, float(expected.reshape(-1)[0]), rtol=0, atol=1e-10)
    margin = np.asarray(estimator.get_booster().predict(DMatrix(transformed), output_margin=True), dtype=float)
    return values, base, margin, verify_additivity(values, base, margin), type(explainer).__name__


def plot_global(values, transformed, mapping, transformed_summary, core_summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "figure.facecolor": "white"}):
        for frame, names, filename, title in [
            (transformed_summary, mapping.set_index("transformed_feature").display_label.to_dict(),
             "shap_global_transformed_bar.png", "Frozen XGBoost: transformed-feature importance"),
            (core_summary, LABELS, "shap_global_core10_bar.png", "Frozen XGBoost: aggregated Core-10 importance")]:
            column = "transformed_feature" if "transformed_feature" in frame else "original_core10_feature"
            ordered = frame.iloc[::-1]
            fig, ax = plt.subplots(figsize=(10.4, max(5.8, len(frame)*0.36+1.4)))
            ax.barh([names[n] for n in ordered[column]], ordered.mean_absolute_shap, color="#24658a")
            ax.set_xlabel("Mean absolute SHAP (raw margin / log-odds)")
            ax.set_title(title, loc="left", pad=16)
            ax.grid(axis="x", alpha=0.18)
            ax.set_axisbelow(True)
            fig.text(0.02, 0.012, f"All development participants (N={len(values):,}); model associations, not causal effects.", fontsize=9, color="#555555")
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(figures / filename, bbox_inches="tight", dpi=300)
            plt.close(fig)
        shap.summary_plot(values, transformed, feature_names=mapping.display_label.tolist(),
                          max_display=18, show=False, plot_size=(10.4, 9.0),
                          rng=np.random.default_rng(SEED))
        fig = plt.gcf()
        fig.axes[0].set_xlabel("SHAP value (raw margin / log-odds)")
        fig.axes[0].set_title("Frozen XGBoost: transformed-feature SHAP distribution", loc="left", pad=18)
        fig.text(0.01, 0.01, "Feature colors use frozen preprocessed values; categorical labels are NHANES codes.", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(figures / "shap_global_beeswarm.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


def run_global_shap(output_dir=XAI_DIR):
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(DEVELOPMENT_PATH.parent.resolve()):
        raise ValueError("Cannot save explanations into frozen processed data")
    inputs = dict(artifact=DEFAULT_ARTIFACT_PATH, metadata=DEFAULT_METADATA_PATH, development=DEVELOPMENT_PATH)
    before = {key: sha256(path) for key, path in inputs.items()}
    protected = {str(p): sha256(p) for p in RESULTS_DIR.rglob("*")
                 if p.is_file() and not p.resolve().is_relative_to(output_dir.resolve())}
    started = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata = load_metadata(DEFAULT_METADATA_PATH)
        pipeline = load_frozen_model(DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH)
        if before["artifact"] != metadata["artifact_sha256"]:
            raise ValueError("Frozen artifact hash mismatch before explanation")
        development = load_development_data()
        preprocessor, estimator = pipeline.named_steps["preprocessing"], pipeline.named_steps["model"]
        transformed = np.asarray(preprocessor.transform(validate_input_columns(development)), dtype=float)
        if not np.isfinite(transformed).all():
            raise ValueError("Frozen preprocessing produced nonfinite model inputs")
        names = preprocessor.get_feature_names_out().tolist()
        if names != metadata["transformed_feature_names"] or transformed.shape != (len(development), 18):
            raise ValueError("Transformed shape/order differs from frozen model metadata")
        mapping = feature_mapping(preprocessor)
        values, base, margin, additivity, explainer_type = explain_transformed(estimator, transformed, names)
        aggregated = aggregate_core10(values, mapping)
        transformed_summary = global_summary(values, names, "transformed_feature")
        core_summary = global_summary(aggregated, CORE_FEATURES, "original_core10_feature")
        # Record the tempting alternative as a diagnostic, NOT the primary importance.
        naive = mapping.assign(mean_absolute_shap=np.abs(values).mean(axis=0)).groupby(
            "original_core10_feature").mean_absolute_shap.sum()
        naive_rank = pd.DataFrame(dict(original_core10_feature=CORE_FEATURES,
            importance=[naive[c] for c in CORE_FEATURES], feature_index=np.arange(10))).sort_values(
                ["importance", "feature_index"], ascending=[False, True], kind="stable")
        naive_rank["rank"] = np.arange(1, 11)
        core_summary["sum_transformed_mean_absolute_shap_NOT_aggregated"] = core_summary.original_core10_feature.map(naive)
        core_summary["rank_if_summing_column_mean_absolute_shap_NOT_aggregated"] = core_summary.original_core10_feature.map(
            naive_rank.set_index("original_core10_feature")["rank"])
        output_dir.mkdir(parents=True, exist_ok=True)
        transformed_summary.to_csv(output_dir / "shap_global_transformed_features.csv", index=False)
        core_summary.to_csv(output_dir / "shap_global_core10.csv", index=False)
        mapping.to_csv(output_dir / "shap_feature_mapping.csv", index=False)
        np.savez_compressed(output_dir / "shap_values_development.npz", shap_values=values,
            base_values=base, expected_value=base[0], seqn=development.seqn.to_numpy(),
            transformed_feature_names=np.asarray(names, dtype=str), core10_shap_values=aggregated,
            core10_feature_names=np.asarray(CORE_FEATURES, dtype=str), raw_margin=margin)
        plot_global(values, transformed, mapping, transformed_summary, core_summary, output_dir)
    after = {key: sha256(path) for key, path in inputs.items()}
    protected_after = {path: sha256(path) for path in protected}
    if before != after or protected != protected_after:
        raise RuntimeError("Frozen artifact/metadata/data or prior results changed during explanation")
    warning_records = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
    for warning in caught:
        warnings.warn(f"Global SHAP: {warning.message}", warning.category, stacklevel=2)
    provenance = dict(model_version=metadata["model_version"], artifact_sha256=after["artifact"],
        artifact_hash_verified=True, metadata_sha256=after["metadata"], development_sha256=after["development"],
        input_paths={key: str(path) for key, path in inputs.items()}, sha256_before=before, sha256_after=after,
        protected_prior_results_sha256_before=protected, protected_prior_results_sha256_after=protected_after,
        sample_n=len(development), sampling="entire development cohort; no sampling", random_seed=SEED,
        positive_n=int(development.diabetes.sum()), prevalence=float(development.diabetes.mean()),
        explainer=f"shap.{explainer_type}", feature_perturbation="tree_path_dependent", approximate=False,
        background="No explicit background dataset; frozen model training-path cover counts only",
        temporal_cohort_accessed=False, fitting_performed=False,
        model_output="raw XGBoost binary margin / log-odds; not probability changes",
        expected_value=float(base[0]), input_feature_count=10, transformed_feature_count=18,
        transformed_feature_names=names, core10_feature_names=CORE_FEATURES,
        categorical_aggregation="Sum signed transformed contributions per participant before absolute value/global averaging",
        global_sd_ddof=0, rank_ties="original fixed model/Core-10 feature order",
        additivity=additivity, aggregation_maximum_absolute_sum_error=float(np.max(np.abs(aggregated.sum(axis=1)-values.sum(axis=1)))),
        package_versions={name: version(name) for name in ("shap", "matplotlib", "xgboost", "numpy", "pandas", "scikit-learn", "joblib")},
        started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(), warnings=warning_records)
    (output_dir / "shap_global_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    return transformed_summary, core_summary, mapping, provenance

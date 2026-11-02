"""Synthetic-profile LIME/SHAP comparison; never temporal validation or model fitting."""

import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.demo_nhanes_cdss import synthetic_profiles
from src.data.preprocessing import CORE_FEATURES
from src.evaluation.nhanes_preprocessing_comparison import DEVELOPMENT_PATH, OUTPUT_DIR
from src.explainability.nhanes_lime_local import (
    KERNEL_WIDTH, NUM_SAMPLES, SEED, STABILITY_SEEDS, NhanesLimeLocal,
    compare_shap_lime, summarize_stability,
)
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, sha256

RESULT_NAMES = {"lime_synthetic_explanations.csv", "shap_lime_agreement.csv", "lime_stability.csv", "lime_provenance.json"}


def run(output_dir=OUTPUT_DIR / "xai"):
    output_dir = Path(output_dir)
    if any(output_dir.resolve().is_relative_to(path.resolve()) for path in
           (DEVELOPMENT_PATH.parents[2], DEFAULT_ARTIFACT_PATH.parents[1])):
        raise ValueError("Cannot save LIME outputs under data or models")
    inputs = dict(artifact=DEFAULT_ARTIFACT_PATH, metadata=DEFAULT_METADATA_PATH, development=DEVELOPMENT_PATH)
    before = {name: sha256(path) for name, path in inputs.items()}
    previous = {str(p): sha256(p) for p in output_dir.rglob("*") if p.is_file() and p.name not in RESULT_NAMES}
    started = datetime.now(timezone.utc).isoformat()
    explanations, agreement, stability, diagnostics = [], [], [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        service = NhanesLimeLocal()
        for profile, patient in synthetic_profiles().items():
            runs = [service.explain_patient(patient, random_seed=seed, num_samples=NUM_SAMPLES) for seed in STABILITY_SEEDS]
            primary = runs[0]
            for row in primary["local_feature_weights"]:
                explanations.append(dict(profile=profile, model_version=primary["model_version"],
                    random_seed=SEED, num_samples=NUM_SAMPLES, predicted_probability=primary["predicted_probability"],
                    local_intercept=primary["local_intercept"], surrogate_prediction=primary["surrogate_prediction"],
                    local_weighted_r2=primary["local_weighted_r2"], surrogate_absolute_error=primary["surrogate_absolute_error"], **row))
            comparison = compare_shap_lime(primary["shap_result"], primary)
            agreement.append(dict(profile=profile, predicted_probability=primary["predicted_probability"], random_seed=SEED,
                **{key: json.dumps(value) if isinstance(value, list) else value for key, value in comparison.items()}))
            stability.extend(dict(profile=profile, **row) for row in summarize_stability(runs))
            diagnostics.extend(dict(profile=profile, random_seed=result["random_seed"],
                local_weighted_r2=result["local_weighted_r2"], surrogate_prediction=result["surrogate_prediction"],
                surrogate_absolute_error=result["surrogate_absolute_error"],
                top3=[r["feature"] for r in result["local_feature_weights"][:3]],
                top5=[r["feature"] for r in result["local_feature_weights"][:5]]) for result in runs)
    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after or previous != {p: sha256(p) for p in previous}:
        raise RuntimeError("Protected inputs or previous explanation results changed")
    if before["artifact"] != service.backend.metadata["artifact_sha256"]:
        raise RuntimeError("Frozen artifact hash differs from metadata")
    frames = [pd.DataFrame(rows) for rows in (explanations, agreement, stability)]
    provenance = dict(
        model_version=service.backend.metadata["model_version"], input_paths={k: str(v) for k, v in inputs.items()},
        sha256_before=before, sha256_after=after, protected_prior_xai_results_sha256=previous,
        temporal_cohort_accessed=False, frozen_model_refit=False, frozen_preprocessing_refit=False,
        lime_surrogate_fitting=True, target_output="frozen pipeline probability for diabetes class 1",
        reference_n=service.reference_n, reference_selection="entire frozen development cohort; no subsampling or outcome use in LIME",
        reference_missingness="existing fitted median/mode imputers transform only",
        core10_features=CORE_FEATURES, category_codes={k: v.tolist() for k, v in service.category_codes.items()},
        category_handling="LIME contiguous indices decoded to exact NHANES codes before frozen-pipeline prediction",
        random_seed=SEED, stability_seeds=list(STABILITY_SEEDS), num_samples=NUM_SAMPLES,
        lime_settings=dict(discretizer="quartile", discretize_continuous=True, feature_selection="none",
                           distance_metric="euclidean", kernel_width=float(KERNEL_WIDTH),
                           kernel="sqrt(exp(-distance**2 / kernel_width**2))", num_features=10,
                           surrogate="LIME default weighted Ridge(alpha=1, fit_intercept=True)",
                           sampling="independent empirical category/bin sampling and inverse-discretization within numeric bins"),
        primary_shap="existing exact tree-path-dependent signed Core-10 raw-margin/log-odds explanation",
        direction_agreement="matching signs with neutral tolerance 1e-10 within top-k intersections; null for empty intersections; all-feature agreement also reported",
        rank_ties="fixed Core-10 order", synthetic_profiles=synthetic_profiles(), diagnostics=diagnostics,
        warnings=[str(w.message) for w in caught], python_version=platform.python_version(),
        package_versions={p: version(p) for p in ("lime", "shap", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib")},
        source_sha256={str(p): sha256(p) for p in (Path(__file__), Path(__file__).parents[1]/"src/explainability/nhanes_lime_local.py")},
        started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat())
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, name in zip(frames, ("lime_synthetic_explanations.csv", "shap_lime_agreement.csv", "lime_stability.csv")):
        frame.to_csv(output_dir / name, index=False)
    (output_dir / "lime_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(frames[1].to_string(index=False))
    return (*frames, provenance)


if __name__ == "__main__":
    run()

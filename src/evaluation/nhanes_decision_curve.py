"""Secondary population-level DCA using existing OOF and frozen probabilities.

No fitting, recalibration, threshold selection, subgroup DCA, or clinical claims.
"""

import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import nhanes_subgroup_audit as subgroup
from src.models.nhanes_phase2_model import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, sha256

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results/nhanes"
DCA_DIR = RESULTS_DIR / "decision_curve"
REFERENCE_ORDER = ("sensitivity_90_reference", "youden_reference", "sensitivity_80_reference", "default")
BASE_GRID = np.arange(10, 501, 5, dtype=float)/1000
NB_TIE_TOLERANCE = 1e-12
DATASET_LABELS = {"development": "Development OOF DCA",
                  "temporal": "Temporal post-hoc DCA / POST-HOC / SECONDARY ANALYSIS"}
TEMPORAL_LR_STATUS = "Omitted: no saved temporal LR probabilities or frozen LR artifact were found; no LR was fitted."


def validate_thresholds(thresholds):
    values = np.asarray(thresholds, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not ((values > 0) & (values < 1)).all():
        raise ValueError("DCA thresholds must be nonempty, finite and strictly between 0 and 1")
    return values


def threshold_grid(metadata):
    references = np.array([metadata["reference_thresholds"][name] for name in REFERENCE_ORDER])
    validate_thresholds(references)
    if not ((references >= BASE_GRID[0]) & (references <= BASE_GRID[-1])).all():
        raise ValueError("Frozen references must be inside the pre-specified primary DCA range")
    return np.unique(np.r_[BASE_GRID, references])


def net_benefit_curve(y_true, probabilities, thresholds):
    """Direct NB = TP/N - FP/N * pt/(1-pt), positives defined by p >= pt."""
    y, p = subgroup.validate_arrays(y_true, probabilities)
    thresholds = validate_thresholds(thresholds)
    if not len(y):
        raise ValueError("DCA requires at least one participant")
    order = np.argsort(p, kind="stable")
    prefix_positive = np.r_[0, np.cumsum(y[order], dtype=np.int64)]
    below = np.searchsorted(p[order], thresholds, side="left")
    fn, tn = prefix_positive[below], below-prefix_positive[below]
    positive_n, n = int(y.sum()), len(y)
    tp, fp = positive_n-fn, n-below-(positive_n-fn)
    prevalence = positive_n/n
    weight = thresholds/(1-thresholds)
    none = np.zeros(len(thresholds), dtype=float)
    all_nb = prevalence-(1-prevalence)*weight
    model_nb = tp/n-fp/n*weight
    # Identical decision strategies must also be identical in floating point.
    model_nb[tp+fp == n] = all_nb[tp+fp == n]
    model_nb[tp+fp == 0] = 0.
    def ratio(a, b):
        return np.divide(a, b, out=np.full(len(thresholds), np.nan), where=np.asarray(b) != 0)
    return pd.DataFrame(dict(threshold=thresholds, n=n, positive_count=positive_n, prevalence=prevalence,
                             tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=ratio(tp, positive_n),
                             specificity=ratio(tn, n-positive_n), net_benefit=model_nb,
                             evaluate_all_net_benefit=all_nb, evaluate_none_net_benefit=none))


def align_saved_oof(development, saved, model):
    """Align optional LR to the already-verified XGBoost OOF IDs/labels/folds."""
    required = {"model", "seqn", "diabetes", "nhanes_cycle", "outer_fold", "probability"}
    if not required <= set(saved):
        raise ValueError(f"Missing saved OOF columns: {sorted(required-set(saved))}")
    selected = saved[saved.model == model]
    if selected.empty:
        return None
    reference = saved[saved.model == "XGBoost"]
    for data in (selected, reference):
        if data.seqn.isna().any() or data.seqn.duplicated().any():
            raise ValueError("OOF participant IDs must be unique/nonmissing within each model")
        if set(data.seqn) != set(development.seqn) or len(data) != len(development):
            raise ValueError("OOF participant IDs must cover exactly the development cohort")
    selected, reference = [data.set_index("seqn").loc[development.seqn] for data in (selected, reference)]
    for field in ("diabetes", "nhanes_cycle", "outer_fold"):
        if not np.array_equal(selected[field].to_numpy(), reference[field].to_numpy()):
            raise ValueError(f"Comparator OOF {field} differs from verified participant-aligned XGBoost OOF")
    y, p = subgroup.validate_arrays(selected.diabetes, selected.probability)
    if not np.array_equal(y, development.diabetes.to_numpy()):
        raise ValueError("OOF outcomes differ from development participants")
    return p


def load_dca_inputs():
    # Reuse the audit's cohort/hash/parameter/fold/ID checks and frozen-only
    # temporal inference. This helper never constructs or fits a new model.
    development, xgb_oof, temporal, xgb_temporal, metadata = subgroup.load_prediction_inputs()
    saved = pd.read_csv(subgroup.OOF_PATH, float_precision="round_trip")
    lr_oof = align_saved_oof(development, saved, "LogisticRegression")
    provenance = json.loads(subgroup.OOF_PROVENANCE_PATH.read_text(encoding="utf-8"))
    if lr_oof is not None and "LogisticRegression" not in provenance["model_parameters"]:
        raise ValueError("Saved LR OOF predictions lack archived model-parameter provenance")
    return {"development": dict(frame=development, xgb=xgb_oof, lr=lr_oof),
            "temporal": dict(frame=temporal, xgb=xgb_temporal, lr=None)}, metadata, {
                "development_lr_available": lr_oof is not None,
                "development_lr_parameters": provenance["model_parameters"].get("LogisticRegression") if lr_oof is not None else None,
                "temporal_lr_available": False, "temporal_lr_status": TEMPORAL_LR_STATUS,
                "search_scope": "models/ and results/: only primary XGBoost joblib, development OOF CSV, and development SHAP NPZ; no temporal LR artifact/prediction set"}


def combined_curve(y, xgb, lr, thresholds, *, dataset, model_version):
    x = net_benefit_curve(y, xgb, thresholds)
    result = x[["threshold", "n", "positive_count", "prevalence", "evaluate_all_net_benefit", "evaluate_none_net_benefit"]].copy()
    result.insert(0, "dataset", dataset)
    result.insert(1, "analysis_type", DATASET_LABELS[dataset])
    result["model_version"], result["lr_available"] = model_version, lr is not None
    for label, probabilities in (("xgb", xgb), ("lr", lr)):
        if probabilities is None:
            result[label+"_net_benefit"] = np.nan
            continue
        curve = x if label == "xgb" else net_benefit_curve(y, probabilities, thresholds)
        for field in ("net_benefit", "tp", "fp", "tn", "fn", "sensitivity", "specificity"):
            result[label+"_"+field] = curve[field].to_numpy()
        d_all = result[label+"_net_benefit"]-result.evaluate_all_net_benefit
        d_none = result[label+"_net_benefit"]-result.evaluate_none_net_benefit
        result[label+"_above_both"] = (d_all > NB_TIE_TOLERANCE) & (d_none > NB_TIE_TOLERANCE)
        result[label+"_below_all"] = d_all < -NB_TIE_TOLERANCE
        result[label+"_below_none"] = d_none < -NB_TIE_TOLERANCE
        result[label+"_ties_all"] = np.abs(d_all) <= NB_TIE_TOLERANCE
        result[label+"_ties_none"] = np.abs(d_none) <= NB_TIE_TOLERANCE
    return result


def sampled_intervals(thresholds, condition):
    """Preserve all disjoint contiguous runs on the evaluated grid only.

    Endpoints are tested values, not estimated continuous crossings or recommended
    operating intervals. No behavior between tested points is inferred.
    """
    thresholds = validate_thresholds(thresholds)
    mask = np.asarray(condition)
    if np.any(np.diff(thresholds) <= 0) or mask.dtype.kind != "b" or mask.shape != thresholds.shape:
        raise ValueError("Intervals require strictly increasing thresholds and matching boolean conditions")
    indices = np.flatnonzero(mask)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1) if len(indices) else []
    return [dict(start_threshold=float(thresholds[run[0]]), end_threshold=float(thresholds[run[-1]]),
                 tested_point_count=len(run),
                 preceding_tested_threshold=float(thresholds[run[0]-1]) if run[0] else None,
                 following_tested_threshold=float(thresholds[run[-1]+1]) if run[-1]+1 < len(thresholds) else None)
            for run in runs]


def reference_table(inputs, metadata):
    rows = []
    thresholds = [metadata["reference_thresholds"][name] for name in REFERENCE_ORDER]
    for dataset, data in inputs.items():
        for label, field in (("XGBoost", "xgb"), ("LogisticRegression", "lr")):
            if data[field] is None:
                continue
            curve = net_benefit_curve(data["frame"].diabetes, data[field], thresholds)
            for name, row in zip(REFERENCE_ORDER, curve.to_dict("records")):
                rows.append(dict(dataset=dataset, analysis_type=DATASET_LABELS[dataset], threshold_name=name,
                                 threshold_value=row["threshold"], model=label, N=row["n"], prevalence=row["prevalence"],
                                 TP=row["tp"], FP=row["fp"], TN=row["tn"], FN=row["fn"],
                                 **{key: row[key] for key in ("sensitivity", "specificity", "net_benefit", "evaluate_all_net_benefit", "evaluate_none_net_benefit")}))
    return pd.DataFrame(rows)


def summarize_curve(curve, references):
    result = dict(n=int(curve.n.iloc[0]), prevalence=float(curve.prevalence.iloc[0]),
                  analysis_type=curve.analysis_type.iloc[0], threshold_points=len(curve), models={})
    for short, model in (("xgb", "XGBoost"), ("lr", "LogisticRegression")):
        if short+"_above_both" not in curve:
            continue
        masks = {"above_both": curve[short+"_above_both"].to_numpy(),
                 "not_superior_to_both": ~curve[short+"_above_both"].to_numpy(),
                 **{name: curve[short+"_"+name].to_numpy() for name in ("below_all", "below_none", "ties_all", "ties_none")}}
        result["models"][model] = dict(
            **{name+"_sampled_intervals": sampled_intervals(curve.threshold, mask) for name, mask in masks.items()},
            comparison_counts={name: int(mask.sum()) for name, mask in masks.items()},
            frozen_reference_values=references[references.model == model].to_dict("records"))
    if "lr_net_benefit" in curve and curve.lr_net_benefit.notna().all():
        difference = curve.xgb_net_benefit-curve.lr_net_benefit
        result["xgb_vs_lr"] = dict(
            xgb_higher_sampled_intervals=sampled_intervals(curve.threshold, (difference > NB_TIE_TOLERANCE).to_numpy()),
            lr_higher_sampled_intervals=sampled_intervals(curve.threshold, (difference < -NB_TIE_TOLERANCE).to_numpy()),
            tied_sampled_intervals=sampled_intervals(curve.threshold, (np.abs(difference) <= NB_TIE_TOLERANCE).to_numpy()))
    return result


def plot_curves(curves, metadata, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        for dataset, full in curves.items():
            for zoomed in (False, True):
                curve = full[full.threshold.between(.05, .20)] if zoomed else full
                fig, ax = plt.subplots(figsize=(9.8, 6.2), layout="constrained")
                ax.plot(curve.threshold, curve.xgb_net_benefit, color="#24658a", linewidth=2, label="XGBoost")
                if curve.lr_net_benefit.notna().all():
                    ax.plot(curve.threshold, curve.lr_net_benefit, color="#b34d34", linewidth=1.8, label="Logistic Regression")
                ax.plot(curve.threshold, curve.evaluate_all_net_benefit, color="#555555", linestyle="--", label="Evaluate-all")
                ax.plot(curve.threshold, curve.evaluate_none_net_benefit, color="#222222", linestyle=":", label="Evaluate-none")
                values = curve[["xgb_net_benefit", "lr_net_benefit", "evaluate_all_net_benefit", "evaluate_none_net_benefit"]].to_numpy()
                low, high = np.nanmin(values), np.nanmax(values)
                pad = max(.01, (high-low)*.12)
                ax.set_ylim(low-pad*.3, high+pad)
                for name in REFERENCE_ORDER[:-1]:
                    threshold = metadata["reference_thresholds"][name]
                    ax.axvline(threshold, color="#888888", alpha=.5, linewidth=.8)
                ax.set_xlim((.05, .20) if zoomed else (.01, .50))
                ax.set_xlabel("Threshold probability")
                ax.set_ylabel("Net benefit")
                ax.grid(axis="y", alpha=.15)
                ax.legend(loc="lower left" if zoomed else "upper right", fontsize=9)
                title = "Development OOF decision curve analysis" if dataset == "development" else "Temporal post-hoc DCA — POST-HOC / SECONDARY ANALYSIS"
                ax.set_title(title + ("\nSecondary zoom: 0.05–0.20 (full-range figure also provided)" if zoomed else "\nFull pre-specified range: 0.01–0.50"), loc="left", fontsize=12)
                ref = metadata["reference_thresholds"]
                fig.supxlabel(f"Vertical references, left to right: 90% ({ref['sensitivity_90_reference']:.6f}), "
                              f"Youden ({ref['youden_reference']:.6f}), 80% ({ref['sensitivity_80_reference']:.6f}).\n"
                              "Assessment/referral assumptions; not prospective clinical benefit or equal benefit across demographic groups.", fontsize=8.5)
                stem = dataset+"_dca"+("_zoomed" if zoomed else "")
                for extension in ("png", "pdf"):
                    fig.savefig(figures_dir/f"{stem}.{extension}", dpi=300)
                plt.close(fig)


def run_dca(output_dir=DCA_DIR):
    output_dir = Path(output_dir)
    if not output_dir.resolve().is_relative_to(DCA_DIR.resolve()):
        raise ValueError("DCA outputs must stay under results/nhanes/decision_curve")
    inputs = dict(artifact=DEFAULT_ARTIFACT_PATH, metadata=DEFAULT_METADATA_PATH,
                  development=subgroup.DEVELOPMENT_PATH, temporal=subgroup.TEMPORAL_PATH,
                  saved_oof=subgroup.OOF_PATH, oof_provenance=subgroup.OOF_PROVENANCE_PATH,
                  temporal_provenance=subgroup.TEMPORAL_PROVENANCE_PATH,
                  subgroup_summary=RESULTS_DIR/"subgroup_audit/subgroup_audit_summary.json")
    protected_sources = [ROOT/"app.py", ROOT/"configs/nhanes_recommendation_rules.json",
                         ROOT/"src/models/nhanes_phase2_model.py", ROOT/"src/decision_support/nhanes_diabetes_cdss.py",
                         ROOT/"src/decision_support/nhanes_recommendations.py", ROOT/"src/evaluation/nhanes_subgroup_audit.py"]
    before = {key: sha256(path) for key, path in inputs.items()}
    source_before = {str(path): sha256(path) for path in protected_sources}
    prior = {str(path): sha256(path) for path in RESULTS_DIR.rglob("*") if path.is_file() and not path.resolve().is_relative_to(DCA_DIR.resolve())}
    started = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions, metadata, comparator = load_dca_inputs()
        grid = threshold_grid(metadata)
        curves = {dataset: combined_curve(data["frame"].diabetes, data["xgb"], data["lr"], grid,
                                         dataset=dataset, model_version=metadata["model_version"])
                  for dataset, data in predictions.items()}
        references = reference_table(predictions, metadata)
        summaries = {dataset: summarize_curve(curve, references[references.dataset == dataset]) for dataset, curve in curves.items()}
        output_dir.mkdir(parents=True, exist_ok=True)
        for dataset, curve in curves.items():
            curve.to_csv(output_dir/f"{dataset}_dca.csv", index=False)
        references.to_csv(output_dir/"dca_reference_thresholds.csv", index=False)
        plot_curves(curves, metadata, output_dir/"figures")
    after = {key: sha256(path) for key, path in inputs.items()}
    if before != after or source_before != {path: sha256(path) for path in source_before} or prior != {path: sha256(path) for path in prior}:
        raise RuntimeError("Frozen inputs, model/application/rules, or prior result files changed during DCA")
    summary = dict(
        analysis_type="SECONDARY POPULATION-LEVEL DECISION CURVE ANALYSIS",
        temporal_warning="Temporal post-hoc DCA / POST-HOC / SECONDARY ANALYSIS after prior evaluation; not a new independent validation experiment.",
        subgroup_context_warning="Prior subgroup audit found substantial age-related threshold variation (temporal sensitivity ~12.5% ages 20–39 vs ~91.47% ages 60+ at the frozen 80% reference). Population DCA does not resolve those differences or establish equal decision benefit.",
        subgroup_audit_document="docs/validation/nhanes_subgroup_performance_audit.md",
        subgroup_specific_dca=False, subgroup_specific_thresholds=False, new_threshold_selected=False,
        model_version=metadata["model_version"], threshold_provenance="Frozen metadata reference_thresholds; all models evaluated at the same threshold probabilities",
        frozen_reference_thresholds=metadata["reference_thresholds"], model_retrained=False,
        preprocessing_refit=False, recalibration=False, model_modified=False,
        formula="TP/N - FP/N * pt/(1-pt)", evaluate_all_formula="prevalence - (1-prevalence) * pt/(1-pt)", evaluate_none_formula="0",
        decision_rule="probability >= pt", grid=dict(start=.01, end=.50, base_step=.005, base_points=len(BASE_GRID),
            total_points=len(grid), exact_reference_values_inserted=True),
        interval_interpretation="Contiguous runs of evaluated grid points only; no claim about untested thresholds or exact continuous crossing points.",
        comparison_tolerance=NB_TIE_TOLERANCE, comparison_policy="Strict superiority exceeds benchmark by >1e-12 NB units; smaller differences are floating-point ties, not statistical equivalence.",
        standardized_net_benefit_included=False, uncertainty_intervals_included=False,
        comparator_availability=comparator, development=summaries["development"], temporal_post_hoc=summaries["temporal"],
        methodological_issues=[
            "DIQ010 physician-diagnosed status is not prospective incident diabetes, biochemical diabetes ascertainment, or a validated need-for-referral endpoint.",
            "Threshold odds are hypothetical assessment/referral false-positive versus true-positive utility assumptions; the grid and sensitivity-derived references were not elicited as clinical preferences.",
            "No specified downstream diagnostic pathway, intervention outcomes, treatment effects, clinician study or prospective clinical benefit is established.",
            "Unweighted cohort-level point estimates; no NHANES survey-design inference, confidence intervals, statistical-superiority tests or model-fitting uncertainty.",
            "Previously selected fixed-config OOF probabilities are preferable to in-sample development scores, but do not remove earlier configuration-selection uncertainty.",
            "Population net benefit does not imply universal usefulness or equal benefit across demographic groups; no subgroup-specific DCA or tuning is done.",
            "LR is available only for development OOF; temporal LR inference is omitted because no valid frozen artifact/prediction set exists."],
        started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
        input_paths={key: str(path) for key, path in inputs.items()}, sha256_before=before, sha256_after=after,
        protected_component_sha256=source_before, protected_prior_result_sha256=prior,
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (Path(__file__), ROOT/"scripts/run_nhanes_decision_curve.py")},
        python_version=platform.python_version(), package_versions={name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib")},
        warnings=[str(w.message) for w in caught])
    owned = [output_dir/name for name in ("development_dca.csv", "temporal_dca.csv", "dca_reference_thresholds.csv")]
    owned += [output_dir/"figures"/f"{dataset}_dca{suffix}.{extension}" for dataset in curves for suffix in ("", "_zoomed") for extension in ("png", "pdf")]
    summary["output_sha256"] = {str(path.relative_to(output_dir)): sha256(path) for path in owned}
    (output_dir/"dca_summary.json").write_text(json.dumps(subgroup._json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(references.to_string(index=False))
    return curves["development"], curves["temporal"], references, summary

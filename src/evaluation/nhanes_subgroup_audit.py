"""Post-hoc transparency audit: saved development OOF + frozen temporal inference.

No training, calibration, threshold selection, or fairness certification.
"""

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.load_nhanes import validate_cohort, validate_cohort_separation
from src.data.preprocessing import CORE_FEATURES
from src.decision_support.prototype_presentation import CATEGORY_LABELS
from src.evaluation.nhanes_preprocessing_comparison import DEVELOPMENT_PATH, OUTPUT_DIR
from src.models.nhanes_phase2_model import (
    DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, load_frozen_model, load_metadata,
    predict_probabilities, sha256,
)

ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_PATH = DEVELOPMENT_PATH.with_name("nhanes_validation_2021_2023.csv")
OOF_PATH = OUTPUT_DIR / "threshold_oof_predictions.csv"
OOF_PROVENANCE_PATH = OUTPUT_DIR / "threshold_analysis_provenance.json"
TEMPORAL_PROVENANCE_PATH = OUTPUT_DIR / "temporal_validation_provenance.json"
AUDIT_DIR = OUTPUT_DIR / "subgroup_audit"
BASE_SEED = 42
BOOTSTRAP_REPLICATES = 1000
MIN_N, MIN_POSITIVES, MIN_NEGATIVES = 100, 30, 30
CI_METRICS = ("roc_auc", "average_precision", "brier_score", "sensitivity", "specificity")
OUTPUT_CSVS = ("development_subgroup_metrics.csv", "temporal_subgroup_metrics.csv",
               "development_disparity_summary.csv", "temporal_disparity_summary.csv")
FIGURE_STEMS = ("subgroup_roc_auc", "subgroup_sensitivity", "subgroup_specificity", "completeness_comparison")
FAMILY_TYPES = {"sex": "demographic", "age": "demographic", "race_ethnicity": "demographic",
                "bmi": "clinical", "completeness": "robustness"}
TEMPORAL_LABEL = "POST-HOC / EXPLORATORY SUBGROUP AUDIT"


def age_groups(values):
    values = pd.to_numeric(pd.Series(values), errors="raise").astype(float)
    if ((values.dropna() < 20) | ~np.isfinite(values.dropna())).any():
        raise ValueError("Age grouping expects finite adult ages >=20 or missing")
    result = pd.Series("Missing age", index=values.index, dtype=object)
    result.loc[values.ge(20) & values.lt(40)] = "20–39"
    result.loc[values.ge(40) & values.lt(60)] = "40–59"
    result.loc[values.ge(60)] = "60+"
    return result


def bmi_groups(values):
    values = pd.to_numeric(pd.Series(values), errors="raise").astype(float)
    if not np.isfinite(values.dropna()).all():
        raise ValueError("BMI grouping expects finite values or missing")
    result = pd.Series("Missing BMI", index=values.index, dtype=object)
    result.loc[values.lt(25)] = "<25"
    result.loc[values.ge(25) & values.lt(30)] = "25–<30"
    result.loc[values.ge(30)] = ">=30"
    return result


def completeness_groups(frame):
    missing = set(CORE_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required Core-10 columns: {sorted(missing)}")
    return pd.Series(np.where(frame[CORE_FEATURES].isna().any(axis=1),
                              ">=1 missing Core-10 input", "Complete Core-10 inputs"), index=frame.index)


def subgroup_definitions(frame):
    definitions = {}
    for field in ("sex", "race_ethnicity"):
        mapping = CATEGORY_LABELS[field]
        observed = set(frame[field].dropna().unique())
        if not observed <= set(mapping):
            raise ValueError(f"Unexpected {field} codes: {observed - set(mapping)}")
        groups = [(str(code), label, frame[field].eq(code).to_numpy()) for code, label in mapping.items()]
        if frame[field].isna().any():
            groups.append(("missing", "Missing " + field, frame[field].isna().to_numpy()))
        definitions[field] = groups
    age, bmi, complete = age_groups(frame.age), bmi_groups(frame.bmi), completeness_groups(frame)
    for family, values, labels in (
        ("age", age, ["20–39", "40–59", "60+"]),
        ("bmi", bmi, ["<25", "25–<30", ">=30"]),
        ("completeness", complete, ["Complete Core-10 inputs", ">=1 missing Core-10 input"]),
    ):
        if values.str.startswith("Missing").any():
            labels = labels + values[values.str.startswith("Missing")].unique().tolist()
        definitions[family] = [(label, label, values.eq(label).to_numpy()) for label in labels]
    for family, groups in definitions.items():
        if not np.all(np.column_stack([mask for _, _, mask in groups]).sum(axis=1) == 1):
            raise ValueError(f"Subgroup family {family} must partition all participant rows exactly once")
    return {key: definitions[key] for key in FAMILY_TYPES}


def validate_arrays(y, probabilities):
    y, p = np.asarray(y, dtype=float), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching one-dimensional binary targets and probabilities")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0,1]")
    return y.astype(int), p


def point_metrics(y, probabilities, threshold):
    y, p = validate_arrays(y, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0,1]")
    n, pos = len(y), int(y.sum())
    pred = p >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    tn, fn = n-pos-fp, pos-tp
    def ratio(a, b):
        return float(a/b) if b else np.nan
    both = 0 < pos < n
    return dict(n=n, positive_count=pos, negative_count=n-pos, prevalence=ratio(pos, n),
                roc_auc=float(roc_auc_score(y, p)) if both else np.nan,
                average_precision=float(average_precision_score(y, p)) if both else np.nan,
                brier_score=float(np.mean((p-y)**2)) if n else np.nan,
                sensitivity=ratio(tp, tp+fn), specificity=ratio(tn, tn+fp), precision=ratio(tp, tp+fp),
                npv=ratio(tn, tn+fn), f1=ratio(2*tp, 2*tp+fp+fn), tp=tp, fp=fp, tn=tn, fn=fn)


def instability_reason(metrics):
    reasons = []
    if metrics["n"] < MIN_N:
        reasons.append(f"N < {MIN_N}")
    if metrics["positive_count"] < MIN_POSITIVES:
        reasons.append(f"positive count < {MIN_POSITIVES}")
    if metrics["negative_count"] < MIN_NEGATIVES:
        reasons.append(f"negative count < {MIN_NEGATIVES}")
    return "; ".join(reasons)


def _weighted_rank_metrics(y_sorted, counts_sorted, starts):
    """Exact tie-aware AUC/AP for resampled row multiplicities; sort only once."""
    positives = np.add.reduceat(counts_sorted*y_sorted, starts)
    negatives = np.add.reduceat(counts_sorted*(1-y_sorted), starts)
    npos, nneg = positives.sum(), negatives.sum()
    if not npos or not nneg:
        return np.nan, np.nan
    tp, fp = np.cumsum(positives), np.cumsum(negatives)
    auc = np.sum(negatives * (tp - positives/2)) / (npos*nneg)
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp) != 0)
    ap = np.sum(positives*precision)/npos
    return float(auc), float(ap)


def bootstrap_intervals(y, probabilities, threshold, *, seed=BASE_SEED, replicates=BOOTSTRAP_REPLICATES):
    """Ordinary participant bootstrap conditional on existing predictions.

    Row multiplicities exactly represent replacement-sampled participant indices.
    Predictions/thresholds are never refitted. One-class draws are excluded for
    AUC/AP, with metric-specific valid-draw counts retained. Unstable groups have
    all CIs suppressed according to the pre-specified count policy.
    """
    y, p = validate_arrays(y, probabilities)
    if not isinstance(replicates, int) or replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    metrics = point_metrics(y, p, threshold)
    reason = instability_reason(metrics)
    result = {f"{m}_ci_{bound}": np.nan for m in CI_METRICS for bound in ("lower", "upper")}
    result.update({f"{m}_bootstrap_valid": 0 for m in CI_METRICS})
    result.update(bootstrap_requested=replicates, bootstrap_computed=0, bootstrap_one_class_draws=0,
                  bootstrap_seed=int(seed), estimates_unstable=bool(reason), instability_reason=reason,
                  ci_status="SUPPRESSED_UNSTABLE_GROUP" if reason else "PERCENTILE_95")
    if reason:
        return result
    order = np.argsort(-p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order]) != 0)+1]
    ys, squared_error = y[order], (p-y)**2
    pos, tp = y == 1, (y == 1) & (p >= threshold)
    neg, tn = y == 0, (y == 0) & (p < threshold)
    rng = np.random.default_rng(seed)
    draws = np.full((replicates, len(CI_METRICS)), np.nan)
    for i in range(replicates):
        counts = np.bincount(rng.integers(0, len(y), size=len(y)), minlength=len(y))
        npos, nneg = counts[pos].sum(), counts[neg].sum()
        if not npos or not nneg:
            result["bootstrap_one_class_draws"] += 1
        draws[i, :2] = _weighted_rank_metrics(ys, counts[order], starts)
        draws[i, 2] = np.sum(counts*squared_error)/len(y)
        if npos:
            draws[i, 3] = counts[tp].sum()/npos
        if nneg:
            draws[i, 4] = counts[tn].sum()/nneg
    result["bootstrap_computed"] = replicates
    for i, metric in enumerate(CI_METRICS):
        valid = draws[:, i][np.isfinite(draws[:, i])]
        result[f"{metric}_bootstrap_valid"] = len(valid)
        if len(valid) >= max(2, int(np.ceil(.95*replicates))):
            lower, upper = np.quantile(valid, [.025, .975], method="linear")
            result[f"{metric}_ci_lower"], result[f"{metric}_ci_upper"] = float(lower), float(upper)
        else:
            result["ci_status"] = "INSUFFICIENT_VALID_BOOTSTRAP_DRAWS"
    return result


def subgroup_seed(base_seed, cohort, family, code):
    return int.from_bytes(hashlib.sha256(f"{base_seed}|{cohort}|{family}|{code}".encode()).digest()[:4], "little")


def analyze_cohort(frame, probabilities, metadata, *, cohort, replicates=BOOTSTRAP_REPLICATES, seed=BASE_SEED):
    threshold = metadata["reference_thresholds"]["sensitivity_80_reference"]
    y, p = validate_arrays(frame.diabetes, probabilities)
    rows = []
    label = "DEVELOPMENT SAVED OOF AUDIT" if cohort == "development" else TEMPORAL_LABEL
    for family, groups in subgroup_definitions(frame).items():
        print(f"{label}: {family}, {replicates} bootstrap replicates", flush=True)
        for code, group_label, mask in groups:
            metrics = point_metrics(y[mask], p[mask], threshold)
            rows.append(dict(cohort=cohort, analysis_label=label, subgroup_family=family,
                             subgroup_type=FAMILY_TYPES[family], group_code=code, group_label=group_label,
                             threshold_name="sensitivity_80_reference", threshold=threshold,
                             prediction_rule="probability >= threshold", model_version=metadata["model_version"],
                             prediction_source="saved 5-fold OOF fixed-configuration XGBoost" if cohort == "development" else "frozen saved pipeline inference; post-hoc",
                             **metrics, **bootstrap_intervals(y[mask], p[mask], threshold,
                                 seed=subgroup_seed(seed, cohort, family, code), replicates=replicates)))
    return pd.DataFrame(rows)


def disparity_summary(metrics):
    rows = []
    for family, group in metrics.groupby("subgroup_family", sort=False):
        for metric in ("roc_auc", "sensitivity", "specificity", "brier_score"):
            finite = group[np.isfinite(group[metric])]
            stable = finite[~finite.estimates_unstable]
            low = finite.loc[finite[metric].idxmin()] if len(finite) else None
            high = finite.loc[finite[metric].idxmax()] if len(finite) else None
            rows.append(dict(cohort=group.cohort.iloc[0], analysis_label=group.analysis_label.iloc[0],
                subgroup_family=family, subgroup_type=group.subgroup_type.iloc[0], metric=metric,
                threshold=float(group.threshold.iloc[0]), finite_group_count=len(finite), stable_group_count=len(stable),
                minimum=float(low[metric]) if low is not None else np.nan,
                maximum=float(high[metric]) if high is not None else np.nan,
                performance_range=float(high[metric]-low[metric]) if len(finite) >= 2 else np.nan,
                minimum_group=low.group_label if low is not None else None,
                maximum_group=high.group_label if high is not None else None,
                minimum_group_unstable=bool(low.estimates_unstable) if low is not None else None,
                maximum_group_unstable=bool(high.estimates_unstable) if high is not None else None,
                stable_groups_performance_range=float(stable[metric].max()-stable[metric].min()) if len(stable) >= 2 else np.nan,
                interpretation="Descriptive performance variation; not a statistical test or fairness certification"))
    return pd.DataFrame(rows)


def load_prediction_inputs():
    """Join saved OOF by participant ID; temporal inference exclusively uses artifact."""
    metadata = load_metadata()
    provenance = json.loads(OOF_PROVENANCE_PATH.read_text(encoding="utf-8"))
    if sha256(DEVELOPMENT_PATH) != metadata["development_csv_sha256"] or sha256(DEVELOPMENT_PATH) != provenance["sha256_after"]:
        raise ValueError("Frozen development file does not match saved OOF provenance")
    if (provenance["features"] != CORE_FEATURES or provenance["class_weight"] != "none"
            or provenance["calibration"] != "none" or provenance["preprocessing"] != "A_median"):
        raise ValueError("Saved OOF feature/weight/calibration specification differs")
    if provenance["model_parameters"]["XGBoost"] != metadata["xgboost_hyperparameters"]:
        raise ValueError("Saved OOF XGBoost parameters differ from the frozen configuration")
    recorded_oof_hash = metadata["prior_result_sha256_before"].get(str(OOF_PATH))
    if recorded_oof_hash is None or sha256(OOF_PATH) != recorded_oof_hash:
        raise ValueError("Saved OOF file hash differs from frozen-model provenance")
    prior_temporal = json.loads(TEMPORAL_PROVENANCE_PATH.read_text(encoding="utf-8"))
    if sha256(TEMPORAL_PATH) != prior_temporal["sha256_after"]["temporal_validation"]:
        raise ValueError("Temporal cohort hash differs from the previously evaluated frozen cohort")
    development, temporal = [pd.read_csv(path, float_precision="round_trip") for path in (DEVELOPMENT_PATH, TEMPORAL_PATH)]
    validate_cohort_separation(development, temporal)
    for data, name, n, pos in ((development, "development", 25580, 3690), (temporal, "validation", 7536, 1070)):
        validate_cohort(data, name)
        if len(data) != n or int(data.diabetes.sum()) != pos:
            raise ValueError(f"Frozen {name} cohort counts differ")
    oof = pd.read_csv(OOF_PATH, float_precision="round_trip")
    oof = oof[oof.model == "XGBoost"].copy()
    if oof.seqn.isna().any() or oof.seqn.duplicated().any():
        raise ValueError("Saved XGBoost OOF predictions must have unique nonmissing IDs")
    matched = development[["seqn", "nhanes_cycle", "diabetes"]].merge(
        oof[["seqn", "nhanes_cycle", "diabetes", "outer_fold", "probability"]], on="seqn",
        how="outer", suffixes=("_cohort", "_oof"), indicator=True, validate="one_to_one", sort=False)
    if not matched._merge.eq("both").all() or len(matched) != len(development):
        raise ValueError("Saved OOF IDs do not cover exactly the development cohort")
    matched = matched.set_index("seqn").loc[development.seqn]
    if not (matched.nhanes_cycle_cohort.eq(matched.nhanes_cycle_oof).all() and matched.diabetes_cohort.eq(matched.diabetes_oof).all()):
        raise ValueError("Saved OOF cycle/target labels do not match participant IDs")
    expected_folds = np.zeros(len(development), dtype=int)
    splitter = StratifiedKFold(provenance["outer_folds"], shuffle=provenance["shuffle"], random_state=provenance["seed"])
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(development)), development.diabetes), 1):
        expected_folds[test] = fold
    if not np.array_equal(matched.outer_fold.to_numpy(), expected_folds):
        raise ValueError("OOF fold IDs do not match the saved deterministic CV design")
    dev_p = matched.probability.to_numpy(dtype=float)
    validate_arrays(development.diabetes, dev_p)
    pipeline = load_frozen_model(DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH)
    temporal_p = predict_probabilities(temporal, pipeline=pipeline)
    return development, dev_p, temporal, temporal_p, metadata


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_audit(output_dir=AUDIT_DIR, *, replicates=BOOTSTRAP_REPLICATES, seed=BASE_SEED):
    output_dir = Path(output_dir)
    # Dedicated output tree: never overwrite baseline, XAI, or frozen audit evidence.
    if not output_dir.resolve().is_relative_to(AUDIT_DIR.resolve()):
        raise ValueError("Audit outputs must stay under results/nhanes/subgroup_audit")
    inputs = dict(artifact=DEFAULT_ARTIFACT_PATH, metadata=DEFAULT_METADATA_PATH, development=DEVELOPMENT_PATH,
                  temporal=TEMPORAL_PATH, saved_oof=OOF_PATH, oof_provenance=OOF_PROVENANCE_PATH,
                  prior_temporal_provenance=TEMPORAL_PROVENANCE_PATH,
                  prior_temporal_metrics=OUTPUT_DIR/"temporal_validation_metrics.csv")
    before = {key: sha256(path) for key, path in inputs.items()}
    protected = {str(p): sha256(p) for p in OUTPUT_DIR.rglob("*") if p.is_file() and not p.resolve().is_relative_to(AUDIT_DIR.resolve())}
    started = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        development, dev_p, temporal, temporal_p, metadata = load_prediction_inputs()
        frames = [analyze_cohort(data, probabilities, metadata, cohort=cohort, replicates=replicates, seed=seed)
                  for data, probabilities, cohort in ((development, dev_p, "development"), (temporal, temporal_p, "temporal"))]
        ranges = [disparity_summary(frame) for frame in frames]
        overall = {name: point_metrics(data.diabetes, p, metadata["reference_thresholds"]["sensitivity_80_reference"])
                   for name, data, p in (("development", development, dev_p), ("temporal_post_hoc", temporal, temporal_p))}
        prior = pd.read_csv(inputs["prior_temporal_metrics"], float_precision="round_trip")
        prior = prior[prior.model == "XGBoost"].iloc[0]
        deltas = {m: overall["temporal_post_hoc"][m]-float(prior[m]) for m in ("roc_auc", "average_precision", "brier_score")}
        if not np.allclose(list(deltas.values()), 0, atol=1e-10, rtol=0):
            raise ValueError(f"Frozen temporal probabilities do not reproduce prior aggregate metrics: {deltas}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame, name in zip(frames+ranges, OUTPUT_CSVS):
            frame.to_csv(output_dir/name, index=False)
        plot_audit(frames[0], frames[1], output_dir/"figures")
    after = {key: sha256(path) for key, path in inputs.items()}
    if before != after or protected != {path: sha256(path) for path in protected}:
        raise RuntimeError("Frozen inputs or previous result files changed during subgroup audit")
    summary = dict(
        title="POST-HOC / EXPLORATORY SUBGROUP PERFORMANCE AUDIT; NOT FAIRNESS CERTIFICATION",
        model_version=metadata["model_version"], started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
        temporal_status=TEMPORAL_LABEL, prior_temporal_evaluation_acknowledged=True,
        development_prediction_source="Previously saved fixed-configuration 5-fold OOF predictions; ID/label/fold/hash/parameter consistency verified",
        temporal_prediction_source="Reapplication of the frozen saved artifact, without fitting; post-hoc/exploratory",
        threshold_name="sensitivity_80_reference", threshold=metadata["reference_thresholds"]["sensitivity_80_reference"],
        threshold_optimization=False, model_fitting=False, preprocessing_fitting=False, calibration=False,
        bootstrap=dict(base_seed=seed, replicates=replicates, confidence_level=.95, method="ordinary participant bootstrap with replacement; percentile 2.5/97.5; fixed predictions",
                       implementation="Resampled row multiplicities; exact tie-aware rank metrics, validated against sklearn",
                       subgroup_seed="first four bytes little-endian SHA256(base_seed|cohort|family|group_code)",
                       one_class_policy="Skip AUC/AP; use metric-specific valid denominators for sensitivity/specificity",
                       minimum_valid_draw_fraction=.95,
                       instability_policy=dict(minimum_n=MIN_N, minimum_positives=MIN_POSITIVES, minimum_negatives=MIN_NEGATIVES,
                                               action="Keep all point estimates/rows; suppress all CIs for groups below any count minimum")),
        grouping=dict(age="[20,40), [40,60), >=60", bmi="<25, [25,30), >=30; missing BMI retained separately",
                      completeness="Any missing ORIGINAL Core-10 value before imputation", subgroup_types=FAMILY_TYPES,
                      sex_mapping=CATEGORY_LABELS["sex"], race_mapping=CATEGORY_LABELS["race_ethnicity"],
                      mapping_source="Existing src/decision_support/prototype_presentation.py CATEGORY_LABELS; RIDRETH3 via src/data/load_nhanes.py"),
        overall_metrics=overall, delta_vs_prior_temporal_metrics=deltas,
        largest_ranges={name: frame.sort_values("performance_range", ascending=False).groupby("metric", sort=False).head(1).to_dict("records")
                        for name, frame in zip(("development", "temporal_post_hoc"), ranges)},
        unstable_groups={name: frame[frame.estimates_unstable][["subgroup_family", "group_label", "n", "positive_count", "negative_count", "instability_reason"]].to_dict("records")
                         for name, frame in zip(("development", "temporal_post_hoc"), frames)},
        sha256_before=before, sha256_after=after, input_paths={key: str(path) for key, path in inputs.items()},
        protected_prior_result_sha256=protected, warnings=[str(w.message) for w in caught],
        python_version=platform.python_version(), package_versions={name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "scipy", "joblib")},
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (Path(__file__), ROOT/"scripts/run_nhanes_subgroup_audit.py", ROOT/"src/decision_support/prototype_presentation.py")},
        limitations=["Unweighted analytic cohorts; IID participant bootstrap is not NHANES complex-survey design inference.",
                     "CIs condition on fixed OOF/temporal predictions and omit fitting, tuning, and fold uncertainty.",
                     "Development OOF fixed configuration was selected in earlier development work; this is not a new fully nested subgroup estimate.",
                     "Pointwise CIs only; no simultaneous intervals, multiplicity correction or inferential test of subgroup differences.",
                     "AP/PPV/NPV depend on prevalence; across-family rows share participants.",
                     "Race/ethnicity is a frozen predictive input and a social category; differences do not establish causation or certify fairness.",
                     "Outcome is physician-diagnosed diabetes status, potentially reflecting detection/access differences; not incident diabetes.",
                     "Temporal results are explicitly post-hoc/exploratory after prior evaluation; not untouched validation."])
    # Hash only outputs owned by this runner, not independently regenerated test
    # reports or user files that happen to share the output directory.
    owned = [output_dir/name for name in OUTPUT_CSVS] + [output_dir/"figures"/f"{stem}.{extension}"
             for stem in FIGURE_STEMS for extension in ("png", "pdf")]
    summary["output_sha256"] = {str(path.relative_to(output_dir)): sha256(path) for path in owned}
    (output_dir/"subgroup_audit_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    return (*frames, *ranges, summary)


def plot_audit(development, temporal, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = {"sex": "Sex (demographic)", "age": "Age (demographic)", "race_ethnicity": "Race/ethnicity (demographic)",
              "bmi": "BMI (clinical subgroup)", "completeness": "Input completeness (robustness)"}
    titles = {"roc_auc": "ROC-AUC", "sensitivity": "Sensitivity at frozen development reference", "specificity": "Specificity at frozen development reference"}
    with plt.rc_context({"font.size": 9, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        for metric, title in titles.items():
            fig, axes = plt.subplots(5, 2, figsize=(14, 18), layout="constrained")
            for column, (frame, cohort_label, color) in enumerate(((development, "Development: saved OOF", "#24658a"),
                                                                  (temporal, "Temporal: POST-HOC / EXPLORATORY", "#b34d34"))):
                for row, family in enumerate(FAMILY_TYPES):
                    group, ax = frame[frame.subgroup_family == family], axes[row, column]
                    for y, record in enumerate(group.to_dict("records")):
                        point, low, high = [record[key] for key in (metric, metric+"_ci_lower", metric+"_ci_upper")]
                        if np.isfinite(low) and np.isfinite(high):
                            ax.hlines(y, low, high, color=color, linewidth=1.5)
                            ax.plot([low, high], [y, y], "|", color=color)
                        if np.isfinite(point):
                            ax.plot(point, y, marker="x" if record["estimates_unstable"] else "o", color="#666666" if record["estimates_unstable"] else color, markersize=5)
                        else:
                            ax.text(.02, y, "undefined", color="#666666", va="center")
                    ax.set_yticks(np.arange(len(group)), [f"{r.group_label}{'*' if r.estimates_unstable else ''}\nN={r.n:,}; diabetes+={r.positive_count:,}" for r in group.itertuples()])
                    ax.invert_yaxis()
                    ax.set_xlim(0, 1)
                    ax.grid(axis="x", alpha=.15)
                    ax.set_title(f"{cohort_label}\n{labels[family]}", loc="left", fontsize=9)
                    ax.set_xlabel(title)
            fig.suptitle(f"Frozen NHANES XGBoost: {title}\nDescriptive subgroup performance; not fairness certification", fontsize=13)
            fig.supxlabel("95% pointwise participant-bootstrap percentile CIs; * CI suppressed by count policy.\n"
                          f"Frozen sensitivity_80_reference = {development.threshold.iloc[0]:.17g}; no subgroup-specific optimization.", fontsize=9)
            for extension in ("png", "pdf"):
                fig.savefig(figures_dir/f"subgroup_{metric}.{extension}", dpi=300)
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")
        for ax, metric, label in zip(axes.flat, ("roc_auc", "average_precision", "brier_score", "sensitivity"),
                                     ("ROC-AUC", "Average precision", "Brier score (lower is better)", "Sensitivity at frozen reference")):
            for offset, frame, color, cohort_label in ((-.06, development, "#24658a", "Development OOF"),
                                                      (.06, temporal, "#b34d34", "Temporal POST-HOC / EXPLORATORY")):
                group = frame[frame.subgroup_family == "completeness"]
                x = np.arange(len(group))+offset
                ax.plot(x, group[metric], "o-", color=color, label=cohort_label)
                for position, record in zip(x, group.to_dict("records")):
                    low, high = record[metric+"_ci_lower"], record[metric+"_ci_upper"]
                    if np.isfinite(low) and np.isfinite(high):
                        ax.vlines(position, low, high, color=color)
                ax.set_xticks([0, 1], ["Complete Core-10", ">=1 missing Core-10"])
                ax.set_ylabel(label)
                ax.grid(axis="y", alpha=.15)
        # A single shared legend avoids obscuring the other panels' CI endpoints.
        axes[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle("Input-completeness robustness comparison\nFrozen-model subgroup audit; not a fairness category", fontsize=13)
        fig.supxlabel("Original input missingness; fitted pipeline handles missing values.\n"
                      f"Temporal POST-HOC / EXPLORATORY; 95% CIs; frozen threshold = {development.threshold.iloc[0]:.17g}.", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(figures_dir/f"completeness_comparison.{extension}", dpi=300)
        plt.close(fig)

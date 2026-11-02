from pathlib import Path
import pandas as pd


NHANES_ROOT = Path(__file__).resolve().parents[2] / "data/raw/nhanes"


CYCLE_CONFIG = {
    "2011_2012": {
        "path": NHANES_ROOT / "2011_2012",
        "files": {
            "demo": "DEMO_G.xpt",
            "diq": "DIQ_G.xpt",
            "bmx": "BMX_G.xpt",
            "bpx": "BPX_G.xpt",
            "hdl": "HDL_G.xpt",
            "tchol": "TCHOL_G.xpt",
            "smq": "SMQ_G.xpt",
            "paq": "PAQ_G.xpt",
        },
        "bp_type": "legacy",
    },

    "2013_2014": {
        "path": NHANES_ROOT / "2013_2014",
        "files": {
            "demo": "DEMO_H.xpt",
            "diq": "DIQ_H.xpt",
            "bmx": "BMX_H.xpt",
            "bpx": "BPX_H.xpt",
            "hdl": "HDL_H.xpt",
            "tchol": "TCHOL_H.xpt",
            "smq": "SMQ_H.xpt",
            "paq": "PAQ_H.xpt",
        },
        "bp_type": "legacy",
    },

    "2015_2016": {
        "path": NHANES_ROOT / "2015_2016",
        "files": {
            "demo": "DEMO_I.xpt",
            "diq": "DIQ_I.xpt",
            "bmx": "BMX_I.xpt",
            "bpx": "BPX_I.xpt",
            "hdl": "HDL_I.xpt",
            "tchol": "TCHOL_I.xpt",
            "smq": "SMQ_I.xpt",
            "paq": "PAQ_I.xpt",
        },
        "bp_type": "legacy",
    },

    "2017_2020": {
        "path": NHANES_ROOT / "2017_March_2020",
        "files": {
            "demo": "P_DEMO.xpt",
            "diq": "P_DIQ.xpt",
            "bmx": "P_BMX.xpt",
            "bpx": "P_BPXO.xpt",
            "hdl": "P_HDL.xpt",
            "tchol": "P_TCHOL.xpt",
            "smq": "P_SMQ.xpt",
            "paq": "P_PAQ.xpt",
        },
        "bp_type": "oscillometric",
    },
}


RENAME_COLUMNS = {'SEQN': 'seqn',
 'RIDAGEYR': 'age',
 'RIAGENDR': 'sex',
 'RIDRETH3': 'race_ethnicity',
 'DIQ010': 'diabetes_status',
 'BMXBMI': 'bmi',
 'BMXWAIST': 'waist_cm',
 'LBDHDD': 'hdl_mg_dl',
 'LBXTC': 'total_cholesterol_mg_dl',
 'SMQ020': 'smoking_lifetime',
 'SMQ040': 'smoking_current',
 'PAQ605': 'vigorous_work',
 'PAQ620': 'moderate_work',
 'PAQ635': 'walk_bicycle',
 'PAQ650': 'vigorous_recreation',
 'PAQ665': 'moderate_recreation'}

DEVELOPMENT_CYCLES = tuple(CYCLE_CONFIG)
VALIDATION_CYCLE = "2021_2023"
VALIDATION_CONFIG = {
    "path": NHANES_ROOT / VALIDATION_CYCLE,
    "files": {key: name + "_L.xpt" for key, name in {
        "demo": "DEMO", "diq": "DIQ", "bmx": "BMX", "bpx": "BPXO",
        "hdl": "HDL", "tchol": "TCHOL", "smq": "SMQ"}.items()},
    "bp_type": "oscillometric",
}


def require_columns(df, columns, context):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def validate_ids(df, column, context):
    require_columns(df, [column], context)
    if df.empty:
        raise ValueError(f"{context}: empty cohort/component")
    if df[column].isna().any() or df[column].duplicated().any():
        raise ValueError(f"{context}: missing or duplicate {column} values")


def read_xpt(path: Path) -> pd.DataFrame:
    """Read without modifying raw files; preserve the existing tiny-zero correction."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing NHANES file: {path}")
    df = pd.read_sas(path, format="xport")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].mask(df[col].abs() < 1e-70, 0.0)
    return df


def harmonize_blood_pressure(df, bp_type):
    """Average the first three readings, retaining zeros and skipping missing readings."""
    if bp_type not in ("legacy", "oscillometric"):
        raise ValueError(f"Unknown BP type: {bp_type}")
    prefix = "BPX" if bp_type == "legacy" else "BPXO"
    systolic = [f"{prefix}SY{i}" for i in (1, 2, 3)]
    diastolic = [f"{prefix}DI{i}" for i in (1, 2, 3)]
    require_columns(df, ["SEQN"] + systolic + diastolic, "blood pressure")
    result = df[["SEQN"]].copy()
    result["mean_systolic_bp"] = df[systolic].mean(axis=1, skipna=True)
    result["mean_diastolic_bp"] = df[diastolic].mean(axis=1, skipna=True)
    return result


def load_cycle(cycle: str) -> pd.DataFrame:
    """Harmonize one cycle; validation is configured separately from development."""
    if cycle == VALIDATION_CYCLE:
        config = VALIDATION_CONFIG
    elif cycle in CYCLE_CONFIG:
        config = CYCLE_CONFIG[cycle]
    else:
        raise ValueError(f"Unknown cycle: {cycle}")
    columns = {
        "demo": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"],
        "diq": ["SEQN", "DIQ010"], "bmx": ["SEQN", "BMXBMI", "BMXWAIST"],
        "hdl": ["SEQN", "LBDHDD"], "tchol": ["SEQN", "LBXTC"],
        "smq": ["SEQN", "SMQ020", "SMQ040"],
        "paq": ["SEQN", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"],
    }
    components = {}
    for name, filename in config["files"].items():
        frame = read_xpt(config["path"] / filename)
        context = f"{cycle}/{filename}"
        validate_ids(frame, "SEQN", context)
        if name == "bpx":
            frame = harmonize_blood_pressure(frame, config["bp_type"])
        else:
            require_columns(frame, columns[name], context)
            frame = frame[columns[name]].copy()
        components[name] = frame
    df = components.pop("demo")
    for frame in components.values():
        df = df.merge(frame, on="SEQN", how="left", validate="one_to_one")
    df = df.rename(columns=RENAME_COLUMNS)
    df["nhanes_cycle"] = cycle
    return df


def load_all_development_cycles() -> pd.DataFrame:
    pooled = pd.concat([load_cycle(c) for c in DEVELOPMENT_CYCLES], ignore_index=True)
    validate_ids(pooled, "seqn", "pooled development")
    return pooled


def validate_cohort(df, cohort):
    from src.data.preprocessing import CORE_FEATURES
    require_columns(df, ["seqn", "nhanes_cycle", "diabetes"] + CORE_FEATURES, cohort)
    validate_ids(df, "seqn", cohort)
    allowed = DEVELOPMENT_CYCLES if cohort == "development" else (VALIDATION_CYCLE,)
    if cohort not in ("development", "validation"):
        raise ValueError(f"Unknown cohort: {cohort}")
    if not df["nhanes_cycle"].isin(allowed).all():
        raise ValueError(f"{cohort}: unexpected survey cycles; cohorts must remain separate")
    if not df["age"].ge(20).all() or not df["diabetes"].isin([0, 1]).all():
        raise ValueError(f"{cohort}: expected adults with binary diabetes outcomes")


def validate_cohort_separation(development, validation):
    validate_cohort(development, "development")
    validate_cohort(validation, "validation")
    if development["seqn"].isin(validation["seqn"]).any():
        raise ValueError("Development and validation SEQN overlap")

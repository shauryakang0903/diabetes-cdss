import warnings
from src.data.load_nhanes import require_columns, validate_ids, DEVELOPMENT_CYCLES, VALIDATION_CYCLE

CORE_FEATURES = ['age', 'sex', 'race_ethnicity', 'bmi', 'waist_cm', 'mean_systolic_bp', 'mean_diastolic_bp', 'hdl_mg_dl', 'total_cholesterol_mg_dl', 'smoking_status']
ACTIVITY_COLUMNS = ['vigorous_work', 'moderate_work', 'walk_bicycle', 'vigorous_recreation', 'moderate_recreation']

def warn_codes(series, allowed):
    unexpected = series[series.notna() & ~series.isin(allowed)].unique()
    if len(unexpected):
        warnings.warn(f"{series.name}: unexpected categorical codes {unexpected.tolist()}", UserWarning, stacklevel=2)

import numpy as np
import pandas as pd


def clean_yes_no(series: pd.Series) -> pd.Series:
    """
    NHANES yes/no questionnaire coding:
    1 = Yes
    2 = No

    Refused / Don't know / invalid values become NaN.

    Returns:
    1.0 = Yes
    0.0 = No
    NaN = unknown / invalid
    """
    warn_codes(series, [1, 2, 7, 9])
    return series.map({
        1.0: 1.0,
        2.0: 0.0,
    })


def derive_smoking_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive smoking status using:
        smoking_lifetime = SMQ020
        smoking_current  = SMQ040

    Result:
        0 = Never smoker
        1 = Former smoker
        2 = Current smoker

    Logic:
        SMQ020 = 2 -> Never smoker

        SMQ020 = 1 and SMQ040 = 3
            -> Former smoker

        SMQ020 = 1 and SMQ040 = 1 or 2
            -> Current smoker

    Anything else remains missing.
    """

    df = df.copy()

    require_columns(df, ["smoking_lifetime", "smoking_current"], "smoking")
    warn_codes(df["smoking_lifetime"], [1, 2, 7, 9])
    warn_codes(df["smoking_current"], [1, 2, 3, 7, 9])
    df["smoking_status"] = np.nan

    # Never smoked 100 cigarettes
    never_mask = df["smoking_lifetime"] == 2.0

    df.loc[
        never_mask,
        "smoking_status"
    ] = 0.0

    # Smoked 100+ cigarettes but currently does not smoke
    former_mask = (
        (df["smoking_lifetime"] == 1.0)
        &
        (df["smoking_current"] == 3.0)
    )

    df.loc[
        former_mask,
        "smoking_status"
    ] = 1.0

    # Smokes every day or some days
    current_mask = (
        (df["smoking_lifetime"] == 1.0)
        &
        (df["smoking_current"].isin([1.0, 2.0]))
    )

    df.loc[
        current_mask,
        "smoking_status"
    ] = 2.0

    return df


def clean_activity_variables(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert NHANES physical-activity yes/no variables:

        1 = Yes -> 1
        2 = No  -> 0

    Refused / Don't know / other unexpected values -> NaN
    """

    df = df.copy()

    activity_columns = [
        "vigorous_work",
        "moderate_work",
        "walk_bicycle",
        "vigorous_recreation",
        "moderate_recreation",
    ]

    require_columns(df, activity_columns, "physical activity")
    for col in activity_columns:
        df[col] = clean_yes_no(df[col])

    return df


def clean_diabetes_outcome(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create binary outcome:

        DIQ010 original:
        1 = Yes
        2 = No

    Borderline, refused, don't know and missing
    are excluded from the primary binary outcome.

    New variable:
        diabetes = 1 -> diagnosed diabetes
        diabetes = 0 -> no diagnosed diabetes
    """

    df = df.copy()

    require_columns(df, ["diabetes_status"], "diabetes outcome")
    warn_codes(df["diabetes_status"], [1, 2, 3, 7, 9])
    df["diabetes"] = df["diabetes_status"].map({
        1.0: 1.0,
        2.0: 0.0,
    })

    return df


def clean_basic_categories(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove unexpected category codes from key categorical fields.

    Sex:
        valid NHANES values are retained as 1 / 2.

    Race/ethnicity:
        values are retained for later categorical encoding.

    We intentionally do not one-hot encode here.
    """

    df = df.copy()

    require_columns(df, ["sex", "race_ethnicity"], "basic categories")
    warn_codes(df["sex"], [1, 2])
    warn_codes(df["race_ethnicity"], [1, 2, 3, 4, 6, 7])
    df.loc[
        ~df["sex"].isin([1.0, 2.0]),
        "sex"
    ] = np.nan

    # RIDRETH3 expected category codes in these cycles.
    valid_race_codes = [
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        7.0,
    ]

    df.loc[
        ~df["race_ethnicity"].isin(valid_race_codes),
        "race_ethnicity"
    ] = np.nan

    return df


def add_data_quality_flags(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add flags that help audit suspicious values.

    These flags DO NOT automatically remove observations.
    """

    df = df.copy()

    require_columns(df, ["mean_diastolic_bp", "mean_systolic_bp", "bmi", "hdl_mg_dl", "total_cholesterol_mg_dl"], "quality flags")
    df["flag_zero_diastolic"] = (
        df["mean_diastolic_bp"] == 0
    )

    df["flag_missing_bmi"] = df["bmi"].isna()

    df["flag_missing_bp"] = (
        df["mean_systolic_bp"].isna()
        |
        df["mean_diastolic_bp"].isna()
    )

    df["flag_missing_lipids"] = (
        df["hdl_mg_dl"].isna()
        |
        df["total_cholesterol_mg_dl"].isna()
    )

    return df


def preprocess_nhanes(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply dataset-level cleaning and feature derivation.

    IMPORTANT:
    This function does NOT:
        - impute missing values
        - scale variables
        - create train/test splits
        - fit models

    Those operations belong inside the modeling pipeline.
    """

    df = df.copy()

    required = ["seqn", "nhanes_cycle", "diabetes_status", "smoking_lifetime", "smoking_current"] + CORE_FEATURES[:-1]
    require_columns(df, required, "NHANES preprocessing")
    validate_ids(df, "seqn", "NHANES preprocessing")
    cycles = set(df["nhanes_cycle"].dropna())
    validation = cycles == {VALIDATION_CYCLE}
    if df["nhanes_cycle"].isna().any() or not (validation or cycles <= set(DEVELOPMENT_CYCLES)):
        raise ValueError("Development and validation cycles must remain separate")
    if validation and any(c in df for c in ACTIVITY_COLUMNS):
        raise ValueError("Physical activity is excluded from temporal validation")
    if not validation:
        require_columns(df, ACTIVITY_COLUMNS, "development activity")

    # Adults only
    df = df[
        df["age"] >= 20
    ].copy()

    # Primary binary diabetes outcome
    df = clean_diabetes_outcome(df)

    # Retain only valid primary outcomes
    df = df[
        df["diabetes"].notna()
    ].copy()

    if df.empty:
        raise ValueError("Empty cohort after adult/outcome filtering")

    # Clean basic categorical fields
    df = clean_basic_categories(df)

    # Smoking
    df = derive_smoking_status(df)

    # Physical activity
    if not validation:
        df = clean_activity_variables(df)

    # Quality-control flags
    df = add_data_quality_flags(df)

    # Put ID and outcome first
    preferred_order = [
        "seqn",
        "nhanes_cycle",
        "diabetes",

        "age",
        "sex",
        "race_ethnicity",

        "bmi",
        "waist_cm",

        "mean_systolic_bp",
        "mean_diastolic_bp",

        "hdl_mg_dl",
        "total_cholesterol_mg_dl",

        "smoking_status",

        "vigorous_work",
        "moderate_work",
        "walk_bicycle",
        "vigorous_recreation",
        "moderate_recreation",

        "flag_zero_diastolic",
        "flag_missing_bmi",
        "flag_missing_bp",
        "flag_missing_lipids",
    ]

    existing = [
        col for col in preferred_order
        if col in df.columns
    ]

    remaining = [
        col for col in df.columns
        if col not in existing
    ]

    df = df[
        existing + remaining
    ]

    return df
from pathlib import Path
import pandas as pd
from src.data.load_nhanes import load_cycle, validate_cohort
from src.data.preprocessing import preprocess_nhanes, CORE_FEATURES

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data/processed/nhanes"


def build_cohort():
    df = preprocess_nhanes(load_cycle("2021_2023"))
    validate_cohort(df, "validation")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_cohort()
    # ============================================================
    # OUTCOME AUDIT
    # ============================================================

    print()
    print("DIABETES OUTCOME")
    print("-" * 72)

    counts = (
        df["diabetes"]
        .value_counts()
        .sort_index()
    )

    print(counts)

    positive = (
        df["diabetes"] == 1.0
    ).sum()

    negative = (
        df["diabetes"] == 0.0
    ).sum()

    print(
        f"\nPositive: {positive:,}"
    )

    print(
        f"Negative: {negative:,}"
    )

    print(
        f"Positive proportion: "
        f"{positive / len(df) * 100:.2f}%"
    )


    # ============================================================
    # MISSINGNESS AUDIT
    # ============================================================

    print()
    print("CORE FEATURE MISSINGNESS")
    print("-" * 72)

    missing_table = pd.DataFrame(
        {
            "missing_n":
                df[CORE_FEATURES]
                .isna()
                .sum(),

            "missing_percent":
                (
                    df[CORE_FEATURES]
                    .isna()
                    .mean()
                    * 100
                ),
        }
    )

    missing_table = (
        missing_table
        .sort_values(
            "missing_percent",
            ascending=False,
        )
    )

    print(
        missing_table.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


    # ============================================================
    # RANGE AUDIT
    # ============================================================

    print()
    print("RANGE / EXTREME-VALUE AUDIT")
    print("-" * 72)

    checks = {
        "age < 20":
            df["age"] < 20,

        "BMI < 10":
            df["bmi"] < 10,

        "BMI > 100":
            df["bmi"] > 100,

        "waist < 40 cm":
            df["waist_cm"] < 40,

        "waist > 200 cm":
            df["waist_cm"] > 200,

        "systolic < 60":
            df["mean_systolic_bp"] < 60,

        "systolic > 250":
            df["mean_systolic_bp"] > 250,

        "diastolic < 20":
            df["mean_diastolic_bp"] < 20,

        "diastolic = 0":
            df["mean_diastolic_bp"] == 0,

        "diastolic > 150":
            df["mean_diastolic_bp"] > 150,

        "HDL < 5":
            df["hdl_mg_dl"] < 5,

        "HDL > 250":
            df["hdl_mg_dl"] > 250,

        "total cholesterol < 50":
            df["total_cholesterol_mg_dl"] < 50,

        "total cholesterol > 900":
            df["total_cholesterol_mg_dl"] > 900,
    }

    for label, mask in checks.items():

        count = (
            mask
            .fillna(False)
            .sum()
        )

        print(
            f"{label:<30}: "
            f"{count:,}"
        )


    # ============================================================
    # NUMERIC SUMMARY
    # ============================================================

    numeric_cols = [
        "age",
        "bmi",
        "waist_cm",
        "mean_systolic_bp",
        "mean_diastolic_bp",
        "hdl_mg_dl",
        "total_cholesterol_mg_dl",
    ]

    print()
    print("NUMERIC SUMMARY")
    print("-" * 72)

    print(
        df[numeric_cols]
        .describe()
        .T
        .round(2)
    )


    # ============================================================
    # DUPLICATE CHECK
    # ============================================================

    print()
    print("DUPLICATE CHECK")
    print("-" * 72)

    duplicates = (
        df["seqn"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate SEQN values: "
        f"{duplicates}"
    )


    if duplicates > 0:
        raise ValueError(
            "Duplicate SEQN values detected "
            "in temporal validation cohort."
        )


    # ============================================================
    # FINAL COLUMN ORDER
    # ============================================================

    final_columns = [
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

        "flag_zero_diastolic",
        "flag_missing_bmi",
        "flag_missing_bp",
        "flag_missing_lipids",

        # Keep originals for reproducibility
        "diabetes_status",
        "smoking_lifetime",
        "smoking_current",
    ]


    df = df[
        final_columns
    ].copy()


    # ============================================================
    # SAVE
    # ============================================================

    csv_output = (
        OUTPUT_DIR /
        "nhanes_validation_2021_2023.csv"
    )

    df.to_csv(
        csv_output,
        index=False
    )


    print(
        f"\nSaved CSV:"
        f"\n{csv_output}"
    )


    # Optional Parquet
    parquet_output = (
        OUTPUT_DIR /
        "nhanes_validation_2021_2023.parquet"
    )

    try:

        df.to_parquet(
            parquet_output,
            index=False
        )

        print(
            f"\nSaved Parquet:"
            f"\n{parquet_output}"
        )

    except ImportError:

        print(
            "\nParquet support unavailable. "
            "CSV was saved successfully."
        )


    # Save missingness separately
    missing_output = (
        OUTPUT_DIR /
        "nhanes_validation_2021_2023_missingness.csv"
    )

    missing_table.to_csv(
        missing_output
    )


    print()
    print("=" * 72)
    print("NHANES 2021–2023 VALIDATION COHORT BUILD COMPLETE")
    print("=" * 72)

    print(
        "\nIMPORTANT:"
        "\nThis cohort is NOT part of model training."
        "\nNo imputation has been performed."
        "\nNo scaling has been performed."
        "\nNo model has been fitted."
        "\nPhysical-activity features were intentionally excluded "
        "because the questionnaire changed in 2021–2023."
    )

if __name__ == "__main__":
    main()

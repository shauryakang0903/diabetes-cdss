from pathlib import Path

import pandas as pd

from src.data.load_nhanes import (
    load_all_development_cycles
)

from src.data.preprocessing import (
    preprocess_nhanes
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data/processed/nhanes"

from src.data.load_nhanes import validate_cohort


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 70)
    print("BUILDING NHANES DEVELOPMENT COHORT")
    print("=" * 70)


    # ---------------------------------------------------------
    # Load harmonized raw cycles
    # ---------------------------------------------------------

    raw = load_all_development_cycles()

    print(
        f"\nRaw pooled participants: "
        f"{len(raw):,}"
    )


    # ---------------------------------------------------------
    # Preprocess
    # ---------------------------------------------------------

    df = preprocess_nhanes(raw)
    validate_cohort(df, "development")

    print(
        f"Final adult cohort with valid outcome: "
        f"{len(df):,}"
    )


    # ---------------------------------------------------------
    # Outcome
    # ---------------------------------------------------------

    print()
    print("DIABETES OUTCOME")
    print("-" * 70)

    outcome_counts = (
        df["diabetes"]
        .value_counts()
        .sort_index()
    )

    print(outcome_counts)

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


    # ---------------------------------------------------------
    # Cycle distribution
    # ---------------------------------------------------------

    print()
    print("PARTICIPANTS BY CYCLE")
    print("-" * 70)

    print(
        df["nhanes_cycle"]
        .value_counts()
        .sort_index()
    )


    # ---------------------------------------------------------
    # Smoking status
    # ---------------------------------------------------------

    print()
    print("SMOKING STATUS")
    print("-" * 70)

    print(
        df["smoking_status"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\n0 = Never smoker"
        "\n1 = Former smoker"
        "\n2 = Current smoker"
    )


    # ---------------------------------------------------------
    # Physical activity
    # ---------------------------------------------------------

    activity_cols = [
        "vigorous_work",
        "moderate_work",
        "walk_bicycle",
        "vigorous_recreation",
        "moderate_recreation",
    ]

    print()
    print("PHYSICAL ACTIVITY VALUE COUNTS")
    print("-" * 70)

    for col in activity_cols:

        print(f"\n{col}")

        print(
            df[col]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )


    # ---------------------------------------------------------
    # Missingness
    # ---------------------------------------------------------

    candidate_features = [
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
    ]

    print()
    print("MISSINGNESS")
    print("-" * 70)

    missing_table = pd.DataFrame({
        "missing_n":
            df[candidate_features]
            .isna()
            .sum(),

        "missing_percent":
            (
                df[candidate_features]
                .isna()
                .mean()
                * 100
            )
    })

    missing_table = (
        missing_table
        .sort_values(
            "missing_percent",
            ascending=False
        )
    )

    print(
        missing_table.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


    # ---------------------------------------------------------
    # Zero diastolic inspection
    # ---------------------------------------------------------

    print()
    print("ZERO DIASTOLIC BP AUDIT")
    print("-" * 70)

    zero_bp = df[
        df["flag_zero_diastolic"]
    ]

    print(
        f"Participants with mean diastolic BP = 0: "
        f"{len(zero_bp):,}"
    )

    if len(zero_bp) > 0:

        print("\nBy NHANES cycle:")

        print(
            zero_bp[
                "nhanes_cycle"
            ]
            .value_counts()
            .sort_index()
        )


    # ---------------------------------------------------------
    # Duplicate check
    # ---------------------------------------------------------

    print()
    print("DUPLICATE CHECK")
    print("-" * 70)

    duplicates = (
        df["seqn"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate SEQN values: "
        f"{duplicates}"
    )


    # ---------------------------------------------------------
    # Numeric distributions
    # ---------------------------------------------------------

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
    print("-" * 70)

    print(
        df[numeric_cols]
        .describe()
        .T
        .round(2)
    )


    # ---------------------------------------------------------
    # Save audit tables
    # ---------------------------------------------------------

    missing_output = (
        OUTPUT_DIR /
        "nhanes_development_missingness.csv"
    )

    missing_table.to_csv(
        missing_output
    )


    cycle_output = (
        OUTPUT_DIR /
        "nhanes_development_cycle_counts.csv"
    )

    df[
        "nhanes_cycle"
    ].value_counts().sort_index().rename(
        "n"
    ).to_csv(
        cycle_output
    )


    # ---------------------------------------------------------
    # Save development cohort
    # ---------------------------------------------------------

    parquet_output = (
        OUTPUT_DIR /
        "nhanes_development_2011_2020.parquet"
    )

    csv_output = (
        OUTPUT_DIR /
        "nhanes_development_2011_2020.csv"
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
            "Install pyarrow later if needed."
        )


    df.to_csv(
        csv_output,
        index=False
    )

    print(
        f"\nSaved CSV:"
        f"\n{csv_output}"
    )


    print()
    print("=" * 70)
    print("NHANES DEVELOPMENT COHORT BUILD COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
        "\nNo imputation has been performed."
        "\nNo scaling has been performed."
        "\nNo train/test split has been performed."
        "\nNo model has been trained."
    )

if __name__ == "__main__":
    main()

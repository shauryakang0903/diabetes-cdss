import pandas as pd
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data/processed/nhanes/nhanes_development_2011_2020.csv"

from src.data.load_nhanes import validate_cohort


def main():
    df = pd.read_csv(PATH)
    validate_cohort(df, "development")

    print("=" * 72)
    print("FINAL NHANES DEVELOPMENT COHORT QUALITY AUDIT")
    print("=" * 72)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")

    # ---------------------------------------------------------
    # Duplicate / ID checks
    # ---------------------------------------------------------

    print("\nDUPLICATES")
    print("-" * 72)

    print(
        "Duplicate SEQN:",
        df["seqn"].duplicated().sum()
    )

    # ---------------------------------------------------------
    # Outcome checks
    # ---------------------------------------------------------

    print("\nOUTCOME")
    print("-" * 72)

    print(
        df["diabetes"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # ---------------------------------------------------------
    # Missingness
    # ---------------------------------------------------------

    features = [
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

    print("\nMISSINGNESS")
    print("-" * 72)

    missing = pd.DataFrame({
        "missing_n": df[features].isna().sum(),
        "missing_pct": df[features].isna().mean() * 100,
    })

    print(
        missing
        .sort_values("missing_pct", ascending=False)
        .round(2)
    )

    # ---------------------------------------------------------
    # Range audit
    #
    # These are audit flags only.
    # Nothing is automatically deleted.
    # ---------------------------------------------------------

    print("\nRANGE / EXTREME-VALUE AUDIT")
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
        print(
            f"{label:<30}: "
            f"{mask.fillna(False).sum():,}"
        )

    # ---------------------------------------------------------
    # Extreme observations by cycle
    # ---------------------------------------------------------

    print("\nLOW DIASTOLIC BP BY CYCLE")
    print("-" * 72)

    low_dbp = df[
        df["mean_diastolic_bp"] < 20
    ]

    print(
        low_dbp["nhanes_cycle"]
        .value_counts()
        .sort_index()
    )

    print("\nZERO DIASTOLIC BP BY CYCLE")
    print("-" * 72)

    zero_dbp = df[
        df["mean_diastolic_bp"] == 0
    ]

    print(
        zero_dbp["nhanes_cycle"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Numeric summaries by cycle
    # ---------------------------------------------------------

    numeric = [
        "age",
        "bmi",
        "waist_cm",
        "mean_systolic_bp",
        "mean_diastolic_bp",
        "hdl_mg_dl",
        "total_cholesterol_mg_dl",
    ]

    print("\nMEDIANS BY CYCLE")
    print("-" * 72)

    print(
        df.groupby("nhanes_cycle")[numeric]
        .median()
        .round(2)
    )

    # ---------------------------------------------------------
    # Outcome proportion by cycle
    # ---------------------------------------------------------

    print("\nDIABETES POSITIVE PROPORTION BY CYCLE")
    print("-" * 72)

    cycle_outcome = (
        df.groupby("nhanes_cycle")["diabetes"]
        .agg(["count", "sum", "mean"])
    )

    cycle_outcome["positive_pct"] = (
        cycle_outcome["mean"] * 100
    )

    print(
        cycle_outcome[
            ["count", "sum", "positive_pct"]
        ].round(2)
    )

    print()
    print("=" * 72)
    print("FINAL AUDIT COMPLETE")
    print("=" * 72)

    print(
        "\nNOTE:"
        "\nThese checks flag unusual observations."
        "\nThey do NOT automatically remove valid extreme measurements."
    )

if __name__ == "__main__":
    main()

from src.data.load_nhanes import load_all_development_cycles


df = load_all_development_cycles()

print()
print("=" * 70)
print("POOLED NHANES DEVELOPMENT COHORT AUDIT")
print("=" * 70)

print(f"\nTotal participants: {len(df):,}")

adult = df[df["age"] >= 20].copy()

print(f"Adults age >= 20: {len(adult):,}")

valid = adult[
    adult["diabetes_status"].isin([1.0, 2.0])
].copy()

print(
    f"Adults with valid diabetes outcome: {len(valid):,}"
)

positive = (
    valid["diabetes_status"] == 1.0
).sum()

negative = (
    valid["diabetes_status"] == 2.0
).sum()

print("\nOutcome distribution:")
print(f"Positive: {positive:,}")
print(f"Negative: {negative:,}")

if len(valid):
    print(
        f"Positive proportion: "
        f"{positive / len(valid) * 100:.2f}%"
    )

print("\nParticipants per cycle:")
print(
    valid["nhanes_cycle"]
    .value_counts()
    .sort_index()
)

print("\nFeature availability:")

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
    "smoking_lifetime",
    "smoking_current",
    "vigorous_work",
    "moderate_work",
    "walk_bicycle",
    "vigorous_recreation",
    "moderate_recreation",
]

for col in features:
    available = valid[col].notna().sum()
    missing = valid[col].isna().sum()
    pct = available / len(valid) * 100

    print(
        f"{col:<28}"
        f"{available:>7,} available | "
        f"{missing:>7,} missing | "
        f"{pct:>6.2f}%"
    )

print("\nDuplicate SEQN check:")
duplicates = df["seqn"].duplicated().sum()
print(f"Duplicate SEQN values: {duplicates}")

print("\nBasic numeric ranges:")

numeric_features = [
    "age",
    "bmi",
    "waist_cm",
    "mean_systolic_bp",
    "mean_diastolic_bp",
    "hdl_mg_dl",
    "total_cholesterol_mg_dl",
]

for col in numeric_features:
    print(
        f"{col:<28}"
        f"min={valid[col].min():>8.2f} "
        f"max={valid[col].max():>8.2f}"
    )

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw/nhanes/2017_March_2020"

FILES = {
    "demo": RAW_DIR / "P_DEMO.xpt",
    "diq": RAW_DIR / "P_DIQ.xpt",
    "bmx": RAW_DIR / "P_BMX.xpt",
    "bpxo": RAW_DIR / "P_BPXO.xpt",
    "hdl": RAW_DIR / "P_HDL.xpt",
    "tchol": RAW_DIR / "P_TCHOL.xpt",
    "smq": RAW_DIR / "P_SMQ.xpt",
    "paq": RAW_DIR / "P_PAQ.xpt",
}


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def load_xpt(path):
    return pd.read_sas(path, format="xport")


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("Loading NHANES files...")

demo = load_xpt(FILES["demo"])
diq = load_xpt(FILES["diq"])
bmx = load_xpt(FILES["bmx"])
bpxo = load_xpt(FILES["bpxo"])
hdl = load_xpt(FILES["hdl"])
tchol = load_xpt(FILES["tchol"])
smq = load_xpt(FILES["smq"])
paq = load_xpt(FILES["paq"])

print("All files loaded successfully.\n")


# ---------------------------------------------------------
# Keep only variables needed for the first audit
# ---------------------------------------------------------

demo = demo[
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
    ]
]

diq = diq[
    [
        "SEQN",
        "DIQ010",
    ]
]

bmx = bmx[
    [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ]
]

bpxo = bpxo[
    [
        "SEQN",
        "BPXOSY1",
        "BPXODI1",
        "BPXOSY2",
        "BPXODI2",
        "BPXOSY3",
        "BPXODI3",
    ]
]

hdl = hdl[
    [
        "SEQN",
        "LBDHDD",
    ]
]

tchol = tchol[
    [
        "SEQN",
        "LBXTC",
    ]
]

smq = smq[
    [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ]
]

paq = paq[
    [
        "SEQN",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
    ]
]


# ---------------------------------------------------------
# Merge
#
# Start with demographics because it contains the full
# participant base for this NHANES release.
# Left joins ensure we do not accidentally discard people
# merely because a component is missing.
# ---------------------------------------------------------

df = demo.copy()

datasets = [
    diq,
    bmx,
    bpxo,
    hdl,
    tchol,
    smq,
    paq,
]

for dataset in datasets:
    df = df.merge(
        dataset,
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

print("Merged dataset shape:")
print(df.shape)
print()


# ---------------------------------------------------------
# Derive average blood pressure
# ---------------------------------------------------------

systolic_cols = [
    "BPXOSY1",
    "BPXOSY2",
    "BPXOSY3",
]

diastolic_cols = [
    "BPXODI1",
    "BPXODI2",
    "BPXODI3",
]

df["mean_systolic_bp"] = df[systolic_cols].mean(
    axis=1,
    skipna=True
)

df["mean_diastolic_bp"] = df[diastolic_cols].mean(
    axis=1,
    skipna=True
)


# ---------------------------------------------------------
# Cohort flow
# ---------------------------------------------------------

print("=" * 60)
print("NHANES COHORT FLOW")
print("=" * 60)

print(f"Total demographic participants: {len(df):,}")

adult = df[df["RIDAGEYR"] >= 20].copy()

print(f"Adults age >= 20: {len(adult):,}")


# Valid primary outcome:
# 1 = Yes
# 2 = No
#
# We temporarily exclude:
# 3 = Borderline
# 7 = Refused
# 9 = Don't know
# Missing
adult_valid_outcome = adult[
    adult["DIQ010"].isin([1.0, 2.0])
].copy()

print(
    "Adults with valid DIQ010 outcome: "
    f"{len(adult_valid_outcome):,}"
)


# ---------------------------------------------------------
# Outcome distribution
# ---------------------------------------------------------

positive = (
    adult_valid_outcome["DIQ010"] == 1.0
).sum()

negative = (
    adult_valid_outcome["DIQ010"] == 2.0
).sum()

print()
print("DIABETES OUTCOME DISTRIBUTION")
print("-" * 60)

print(f"Diabetes positive: {positive:,}")
print(f"Diabetes negative: {negative:,}")

if len(adult_valid_outcome) > 0:
    prevalence = (
        positive /
        len(adult_valid_outcome)
    ) * 100

    print(
        f"Observed positive proportion: "
        f"{prevalence:.2f}%"
    )


# ---------------------------------------------------------
# Feature availability
# ---------------------------------------------------------

print()
print("=" * 60)
print("FEATURE AVAILABILITY")
print("=" * 60)

audit_features = {
    "BMI": "BMXBMI",
    "Waist circumference": "BMXWAIST",
    "Mean systolic BP": "mean_systolic_bp",
    "Mean diastolic BP": "mean_diastolic_bp",
    "HDL cholesterol": "LBDHDD",
    "Total cholesterol": "LBXTC",
    "Lifetime smoking response": "SMQ020",
    "Vigorous work activity": "PAQ605",
    "Moderate work activity": "PAQ620",
    "Walking/cycling": "PAQ635",
    "Vigorous recreation": "PAQ650",
    "Moderate recreation": "PAQ665",
}

for label, col in audit_features.items():

    available = (
        adult_valid_outcome[col]
        .notna()
        .sum()
    )

    missing = (
        adult_valid_outcome[col]
        .isna()
        .sum()
    )

    pct_available = (
        available /
        len(adult_valid_outcome)
        * 100
    )

    print(
        f"{label:<28}"
        f" Available: {available:>6,}"
        f" | Missing: {missing:>6,}"
        f" | {pct_available:>6.2f}% available"
    )


# ---------------------------------------------------------
# Complete-case diagnostic only
#
# IMPORTANT:
# This is NOT our final modeling cohort.
# We are only measuring how many people would remain if
# every core continuous variable had to be present.
# ---------------------------------------------------------

core_complete_cols = [
    "BMXBMI",
    "BMXWAIST",
    "mean_systolic_bp",
    "mean_diastolic_bp",
    "LBDHDD",
    "LBXTC",
]

complete_case = adult_valid_outcome.dropna(
    subset=core_complete_cols
)

print()
print("=" * 60)
print("COMPLETE-CASE DIAGNOSTIC")
print("=" * 60)

print(
    "Participants with all selected core "
    f"continuous variables available: {len(complete_case):,}"
)

print()
print(
    "NOTE: This is only an audit. "
    "No observations are permanently removed yet."
)

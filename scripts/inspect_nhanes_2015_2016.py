from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw/nhanes/2015_2016"

FILES = {
    "DEMO": RAW_DIR / "DEMO_I.xpt",
    "DIQ": RAW_DIR / "DIQ_I.xpt",
    "BMX": RAW_DIR / "BMX_I.xpt",
    "BPX": RAW_DIR / "BPX_I.xpt",
    "HDL": RAW_DIR / "HDL_I.xpt",
    "TCHOL": RAW_DIR / "TCHOL_I.xpt",
    "SMQ": RAW_DIR / "SMQ_I.xpt",
    "PAQ": RAW_DIR / "PAQ_I.xpt",
}

EXPECTED = {
    "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"],
    "DIQ": ["SEQN", "DIQ010"],
    "BMX": ["SEQN", "BMXBMI", "BMXWAIST"],
    "BPX": [
        "SEQN",
        "BPXSY1", "BPXDI1",
        "BPXSY2", "BPXDI2",
        "BPXSY3", "BPXDI3",
    ],
    "HDL": ["SEQN", "LBDHDD"],
    "TCHOL": ["SEQN", "LBXTC"],
    "SMQ": ["SEQN", "SMQ020", "SMQ040"],
    "PAQ": [
        "SEQN",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
    ],
}

print("=" * 70)
print("NHANES 2015–2016 FILE VERIFICATION")
print("=" * 70)

for name, path in FILES.items():
    print(f"\n{name}")
    print("-" * 70)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    df = pd.read_sas(path, format="xport")

    print(f"File: {path.name}")
    print(f"Shape: {df.shape}")
    print(f"SEQN unique: {df['SEQN'].is_unique}")

    print("\nExpected variables:")

    for col in EXPECTED[name]:
        status = "FOUND" if col in df.columns else "MISSING"
        print(f"  {col:<12} {status}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

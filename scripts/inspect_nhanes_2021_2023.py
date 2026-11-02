from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw/nhanes/2021_2023"

FILES = {
    "DEMO": RAW_DIR / "DEMO_L.xpt",
    "DIQ": RAW_DIR / "DIQ_L.xpt",
    "BMX": RAW_DIR / "BMX_L.xpt",
    "BPXO": RAW_DIR / "BPXO_L.xpt",
    "HDL": RAW_DIR / "HDL_L.xpt",
    "TCHOL": RAW_DIR / "TCHOL_L.xpt",
    "SMQ": RAW_DIR / "SMQ_L.xpt",
    "PAQ": RAW_DIR / "PAQ_L.xpt",
}

EXPECTED = {
    "DEMO": [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
    ],
    "DIQ": [
        "SEQN",
        "DIQ010",
    ],
    "BMX": [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ],
    "BPXO": [
        "SEQN",
        "BPXOSY1",
        "BPXODI1",
        "BPXOSY2",
        "BPXODI2",
        "BPXOSY3",
        "BPXODI3",
    ],
    "HDL": [
        "SEQN",
        "LBDHDD",
    ],
    "TCHOL": [
        "SEQN",
        "LBXTC",
    ],
    "SMQ": [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ],
    "PAQ": [
        "SEQN",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
    ],
}

print("=" * 72)
print("NHANES 2021–2023 FILE VERIFICATION")
print("=" * 72)

all_ok = True

for name, path in FILES.items():

    print(f"\n{name}")
    print("-" * 72)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        all_ok = False
        continue

    try:
        df = pd.read_sas(path, format="xport")
    except Exception as exc:
        print(f"ERROR READING FILE: {exc}")
        all_ok = False
        continue

    print(f"File: {path.name}")
    print(f"Shape: {df.shape}")

    if "SEQN" in df.columns:
        print(f"SEQN unique: {df['SEQN'].is_unique}")
    else:
        print("SEQN unique: CANNOT CHECK")
        all_ok = False

    print("\nExpected variables:")

    for col in EXPECTED[name]:
        if col in df.columns:
            print(f"  {col:<12} FOUND")
        else:
            print(f"  {col:<12} MISSING")
            all_ok = False

    # Extra visibility for PAQ/BP changes
    if name in {"PAQ", "BPXO"}:
        print("\nAll columns:")
        print(df.columns.tolist())

print("\n" + "=" * 72)

if all_ok:
    print("RESULT: ALL REQUIRED 2021–2023 FILES AND VARIABLES VERIFIED")
else:
    print("RESULT: ONE OR MORE FILES / VARIABLES NEED ATTENTION")

print("=" * 72)

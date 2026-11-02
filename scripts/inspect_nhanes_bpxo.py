from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'data/raw/nhanes/2017_March_2020/P_BPXO.xpt'

df = pd.read_sas(path, format="xport")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

important_cols = [
    "SEQN",
    "BPXOSY1", "BPXODI1",
    "BPXOSY2", "BPXODI2",
    "BPXOSY3", "BPXODI3"
]

print("\nImportant variables:")
for col in important_cols:
    print(f"{col}: {'FOUND' if col in df.columns else 'MISSING'}")

print("\nSEQN unique:", df["SEQN"].is_unique)

existing = [c for c in important_cols if c in df.columns]

print("\nMissing values:")
print(df[existing].isna().sum())

print("\nFirst 5 rows:")
print(df[existing].head())
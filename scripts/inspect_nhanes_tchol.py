from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'data/raw/nhanes/P_TCHOL.xpt'

df = pd.read_sas(path, format="xport")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSEQN unique:", df["SEQN"].is_unique)

print("\nMissing values:")
for col in df.columns:
    if col != "SEQN":
        print(f"{col}: {df[col].isna().sum()}")

print("\nFirst 5 rows:")
print(df.head())
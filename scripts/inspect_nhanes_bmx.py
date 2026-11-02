from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'data/raw/nhanes/P_BMX.xpt'

df = pd.read_sas(path, format="xport")

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

important_cols = ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]

print("\nImportant variables:")
for col in important_cols:
    print(f"{col}: {'FOUND' if col in df.columns else 'MISSING'}")

print("\nSEQN unique:", df["SEQN"].is_unique)

print("\nMissing values in important variables:")
print(df[important_cols].isna().sum())
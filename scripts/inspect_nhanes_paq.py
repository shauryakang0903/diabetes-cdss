from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'data/raw/nhanes/2017_March_2020/P_PAQ.xpt'

df = pd.read_sas(path, format="xport")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSEQN unique:", df["SEQN"].is_unique)

candidate_cols = [
    col for col in df.columns
    if col.startswith("PAQ")
]

print("\nPAQ variables found:")
print(candidate_cols)

print("\nMissing values for PAQ variables:")
print(df[candidate_cols].isna().sum())

print("\nFirst 5 rows:")
print(df[["SEQN"] + candidate_cols].head())
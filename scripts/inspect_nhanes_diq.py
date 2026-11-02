from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'data/raw/nhanes/2017_March_2020/P_DIQ.xpt'

df = pd.read_sas(path, format="xport")

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDIQ010 value counts:")
print(df["DIQ010"].value_counts(dropna=False).sort_index())

print("\nSEQN unique:", df["SEQN"].is_unique)
from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'data/raw/nhanes/2017_March_2020/P_SMQ.xpt'

df = pd.read_sas(path, format="xport")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

important_cols = ["SEQN", "SMQ020", "SMQ040"]

print("\nImportant variables:")
for col in important_cols:
    print(f"{col}: {'FOUND' if col in df.columns else 'MISSING'}")

print("\nSEQN unique:", df["SEQN"].is_unique)

print("\nValue counts:")
for col in ["SMQ020", "SMQ040"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).sort_index())
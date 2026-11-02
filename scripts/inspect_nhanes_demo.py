from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'data/raw/nhanes/2017_March_2020/P_DEMO.xpt'

df = pd.read_sas(path, format="xport")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nSEQN unique:", df["SEQN"].is_unique)
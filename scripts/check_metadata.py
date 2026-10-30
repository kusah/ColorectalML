
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
metadata_file = root / "data" / "processed" / "sample_metadata.csv"

df = pd.read_csv(metadata_file)

print("Metadata columns:")
print(df.columns.tolist())

print("\nSample metadata:")
print(df.to_string(index=False))

print("\nUnique sample titles:")
print(df["title"].value_counts(dropna=False).to_string())

print("\nUnique sample sources:")
print(df["source"].value_counts(dropna=False).to_string())

print("\nCharacteristics:")
print(df["characteristics"].value_counts(dropna=False).to_string())

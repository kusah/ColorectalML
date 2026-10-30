
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

df = pd.read_csv(DATA_DIR / "sample_metadata.csv")

# Recreate patient IDs and groups from the original metadata
def extract_patient_id(text):
    match = re.search(
        r"Patient ID:\s*([A-Za-z0-9_-]+)",
        str(text),
        flags=re.IGNORECASE
    )
    return match.group(1) if match else None

def assign_group(row):
    description = (
        str(row["title"]) + " " +
        str(row["source"]) + " " +
        str(row["characteristics"])
    ).lower()

    if "colorectal adenoma" in str(row["source"]).lower():
        return "Adenoma"
    if "normal mucosa" in str(row["source"]).lower():
        return "Normal"
    if "tissue: adenoma" in description:
        return "Adenoma"
    if "tissue: normal" in description:
        return "Normal"
    return "Unknown"

df["patient_id"] = df["characteristics"].apply(extract_patient_id)
df["group"] = df.apply(assign_group, axis=1)

if df["patient_id"].isna().any():
    print("WARNING: Some patient IDs could not be extracted.")
    print(df.loc[df["patient_id"].isna()].to_string(index=False))

print("Group counts:")
print(df["group"].value_counts(dropna=False))

print("\nPatients with more than two samples:")
counts = df.groupby("patient_id")["sample_id"].count()
repeated = counts[counts > 2].index.tolist()
print(repeated)

for patient in repeated:
    print(f"\n{'=' * 65}")
    print(f"PATIENT: {patient}")
    print(
        df.loc[
            df["patient_id"] == patient,
            ["sample_id", "title", "source", "characteristics", "group"]
        ].to_string(index=False)
    )

print("\n=== All patient/sample assignments ===")
print(
    df[["sample_id", "patient_id", "group"]]
    .sort_values(["patient_id", "group", "sample_id"])
    .to_string(index=False)
)

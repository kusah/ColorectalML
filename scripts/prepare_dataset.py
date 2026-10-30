
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

expression_file = DATA_DIR / "expression_matrix.csv"
metadata_file = DATA_DIR / "sample_metadata.csv"

# Load downloaded data
expression = pd.read_csv(expression_file, index_col=0)
metadata = pd.read_csv(metadata_file)

required = {"sample_id", "title", "source", "characteristics"}
if not required.issubset(metadata.columns):
    raise ValueError(f"Metadata must contain: {required}")

# Assign labels from documented sample descriptions
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

metadata["group"] = metadata.apply(assign_group, axis=1)

# Extract patient identifiers from sample characteristics
def extract_patient_id(text):
    match = re.search(
        r"patient\s*#\s*(\d+)",
        str(text),
        flags=re.IGNORECASE
    )
    return match.group(1) if match else None


metadata["patient_id"] = metadata["title"].apply(
    extract_patient_id
)

# Validate labels before creating a training dataset
print("Group counts:")
print(metadata["group"].value_counts(dropna=False))

if (metadata["group"] == "Unknown").any():
    unknown = metadata.loc[
        metadata["group"] == "Unknown", "sample_id"
    ].tolist()
    raise ValueError(f"Unlabelled samples: {unknown}")

if metadata["patient_id"].isna().any():
    missing = metadata.loc[
        metadata["patient_id"].isna(), "sample_id"
    ].tolist()
    raise ValueError(f"Patient IDs missing for: {missing}")

if metadata["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample IDs in metadata.")

# Convert to samples x probes
X = expression.T
X.index.name = "sample_id"

# Align expression samples with metadata
missing_samples = set(metadata["sample_id"]) - set(X.index)
if missing_samples:
    raise ValueError(
        f"Samples missing from expression matrix: {missing_samples}"
    )

X = X.loc[metadata["sample_id"]]

# Check expression data
if X.isna().all(axis=1).any():
    raise ValueError("At least one sample has no usable expression values.")

if not X.apply(pd.to_numeric, errors="coerce").notna().all().all():
    raise ValueError("Expression matrix contains non-numeric or missing values.")

X = X.apply(pd.to_numeric)

# Save features and labels separately
X.to_csv(DATA_DIR / "X_expression.csv")

metadata[
    ["sample_id", "patient_id", "group"]
].to_csv(
    DATA_DIR / "sample_labels.csv",
    index=False
)

# Save a combined file for inspection only
combined = X.copy()
combined["group"] = metadata.set_index(
    "sample_id"
).loc[X.index, "group"].values
combined["patient_id"] = metadata.set_index(
    "sample_id"
).loc[X.index, "patient_id"].values

combined.to_csv(DATA_DIR / "ml_dataset.csv")

# Check that patients have both tissue types
pair_check = metadata.pivot_table(
    index="patient_id",
    columns="group",
    values="sample_id",
    aggfunc="count",
    fill_value=0
)

print("\nPatient pairing check:")
print(pair_check)

complete_pairs = (
    (pair_check.get("Normal", 0) > 0) &
    (pair_check.get("Adenoma", 0) > 0)
).sum()

print(f"\nPatients represented in both groups: {complete_pairs}")
print("Expression feature matrix:", X.shape)
print("Files saved to:", DATA_DIR)
print("Dataset preparation completed.")

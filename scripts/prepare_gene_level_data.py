
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMMON_DIR = ROOT / "data" / "processed" / "common_genes"
PROCESSED_DIR = ROOT / "data" / "processed"

# Load aligned gene expression matrices
train_expr = pd.read_csv(
    COMMON_DIR / "GSE8671_common_genes.csv",
    index_col=0
)

external_expr = pd.read_csv(
    COMMON_DIR / "GSE50114_common_genes.csv",
    index_col=0
)

# Load original training labels
labels_path = PROCESSED_DIR / "sample_labels.csv"
labels = pd.read_csv(labels_path)

required = {"sample_id", "group"}
if not required.issubset(labels.columns):
    raise ValueError(
        f"Expected columns {required}; found {labels.columns.tolist()}"
    )

# Convert to samples x genes
X_train = train_expr.T
X_external = external_expr.T

# Align training labels to training expression sample order
labels["sample_id"] = labels["sample_id"].astype(str)
X_train.index = X_train.index.astype(str)
X_external.index = X_external.index.astype(str)

labels = labels.drop_duplicates(subset=["sample_id"])

missing_labels = set(X_train.index) - set(labels["sample_id"])
if missing_labels:
    raise ValueError(
        f"Missing labels for training samples: {sorted(missing_labels)}"
    )

labels = labels.set_index("sample_id").loc[X_train.index]

y_train = labels["group"].map({
    "Normal": 0,
    "Adenoma": 1
})

if y_train.isna().any():
    raise ValueError(
        f"Unexpected training labels: {labels['group'].unique()}"
    )

# Verify gene order
if list(X_train.columns) != list(X_external.columns):
    raise ValueError("Gene columns are not aligned.")

# Save datasets
OUT_DIR = PROCESSED_DIR / "gene_level"
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_train.to_csv(OUT_DIR / "X_train.csv", index_label="sample_id")
y_train.rename("label").to_csv(
    OUT_DIR / "y_train.csv", index_label="sample_id"
)
X_external.to_csv(
    OUT_DIR / "X_external.csv", index_label="sample_id"
)

print("Training features:", X_train.shape)
print("External features:", X_external.shape)
print("\nTraining label counts:")
print(y_train.value_counts().rename(index={
    0: "Normal",
    1: "Adenoma"
}))
print("\nExternal missing values:", int(X_external.isna().sum().sum()))
print("Gene order aligned:", list(X_train.columns) == list(X_external.columns))
print("\nSaved to:", OUT_DIR)

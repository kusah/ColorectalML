
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"

print("=== 1. Model metrics ===")
metrics = pd.read_csv(RESULTS / "model_comparison.csv")
print(metrics.to_string(index=False))

print("\n=== 2. Actual vs predicted counts ===")
for name in [
    "logistic_regression",
    "svm",
    "random_forest"
]:
    path = RESULTS / f"{name}_predictions.csv"
    df = pd.read_csv(path)

    print(f"\n{name}")
    print(pd.crosstab(
        df["actual"],
        df["predicted"],
        margins=True
    ))
    print("Prediction counts:")
    print(df["predicted"].value_counts())
    print(
        "All probabilities finite:",
        np.isfinite(df["adenoma_probability"]).all()
    )

print("\n=== 3. Expression matrix checks ===")
X = pd.read_csv(DATA / "X_expression.csv", index_col=0)
labels = pd.read_csv(DATA / "sample_labels.csv")

print("Shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))
print("Infinite values:", int(np.isinf(X.to_numpy()).sum()))
print("Duplicate samples:", int(X.index.duplicated().sum()))
print("Duplicate probes:", int(X.columns.duplicated().sum()))
print("Unique expression rows:", X.drop_duplicates().shape[0])

print("\n=== 4. Patient-group checks ===")
print(
    labels.groupby("patient_id")
    .agg(
        sample_count=("sample_id", "count"),
        class_count=("group", "nunique")
    )
    .to_string()
)

print("\nDiagnostics completed.")


from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

X = pd.read_csv(DATA_DIR / "X_expression.csv", index_col=0)
labels = pd.read_csv(DATA_DIR / "sample_labels.csv")

labels = labels.set_index("sample_id").loc[X.index]

print("=== Repeated patient samples ===")

for patient_id, group in labels.groupby("patient_id"):
    if len(group) <= 2:
        continue

    sample_ids = group.index.tolist()

    print(f"\nPatient: {patient_id}")
    print(group[["group"]].to_string())
    
    for tissue in ["Normal", "Adenoma"]:
        ids = group[group["group"] == tissue].index.tolist()

        if len(ids) > 1:
            values = X.loc[ids].to_numpy(dtype=float)

            correlation = np.corrcoef(values)[0, 1]
            identical = np.array_equal(values[0], values[1])

            print(
                f"{tissue}: {ids}; "
                f"correlation={correlation:.6f}; "
                f"identical={identical}"
            )

print("\nReplicate check completed.")

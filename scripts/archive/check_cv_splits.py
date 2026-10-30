
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

X = pd.read_csv(DATA_DIR / "X_expression.csv", index_col=0)
labels = pd.read_csv(DATA_DIR / "sample_labels.csv").set_index("sample_id")
labels = labels.loc[X.index]

y = labels["group"].map({"Normal": 0, "Adenoma": 1}).to_numpy()
groups = labels["patient_id"].to_numpy()

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Total samples:", len(y))
print("Unique patients:", len(np.unique(groups)))

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    overlap = train_groups & test_groups

    print(f"\nFold {fold}")
    print("Train samples:", len(train_idx))
    print("Test samples:", len(test_idx))
    print(
        "Train classes:",
        dict(zip(*np.unique(y[train_idx], return_counts=True)))
    )
    print(
        "Test classes:",
        dict(zip(*np.unique(y[test_idx], return_counts=True)))
    )
    print("Patient overlap:", overlap)

    assert len(overlap) == 0, "Patient leakage detected!"
    assert len(np.unique(y[train_idx])) == 2, "Training fold lacks a class"
    assert len(np.unique(y[test_idx])) == 2, "Test fold lacks a class"

print("\nAll folds passed the basic checks.")


from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

X_df = pd.read_csv(DATA_DIR / "X_expression.csv", index_col=0)
labels = pd.read_csv(DATA_DIR / "sample_labels.csv").set_index("sample_id")
labels = labels.loc[X_df.index]

X = X_df.to_numpy(dtype=np.float64)
y = labels["group"].map({"Normal": 0, "Adenoma": 1}).to_numpy()
groups = labels["patient_id"].to_numpy()

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("select", SelectKBest(score_func=f_classif, k=100)),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

# Original grouped CV performance
pred = cross_val_predict(
    clone(pipeline), X, y,
    groups=groups, cv=cv, method="predict"
)
print("Original grouped-CV accuracy:", accuracy_score(y, pred))

# Permute labels at the patient level: all samples from one patient
# receive the same shuffled label pattern, preserving pairing structure.
# This is a diagnostic, not a definitive statistical test.
rng = np.random.default_rng(42)
patient_table = labels.reset_index().groupby("patient_id")["group"].first()
patient_ids = patient_table.index.to_numpy()
patient_groups = patient_table.to_numpy()

scores = []

for i in range(30):
    shuffled_patient_labels = rng.permutation(patient_groups)
    mapping = dict(zip(patient_ids, shuffled_patient_labels))
    y_perm = np.array([
        1 if mapping[p] == "Adenoma" else 0
        for p in groups
    ])

    # This preserves each patient's within-patient label pattern only
    # if the data are paired as assumed. Therefore, interpret cautiously.
    try:
        pred_perm = cross_val_predict(
            clone(pipeline), X, y_perm,
            groups=groups, cv=cv, method="predict"
        )
        scores.append(accuracy_score(y_perm, pred_perm))
    except ValueError as exc:
        print(f"Permutation {i + 1} skipped: {exc}")

print("\nPermutation accuracy scores:")
print(np.round(scores, 3))
if scores:
    print("Mean permutation accuracy:", round(float(np.mean(scores)), 3))
    print("Maximum permutation accuracy:", round(float(np.max(scores)), 3))

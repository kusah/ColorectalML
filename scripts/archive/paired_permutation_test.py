
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
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

X_df = pd.read_csv(DATA_DIR / "X_expression.csv", index_col=0)
labels = pd.read_csv(DATA_DIR / "sample_labels.csv").set_index("sample_id")
labels = labels.loc[X_df.index]

X = X_df.to_numpy(dtype=float)
y = labels["group"].map({"Normal": 0, "Adenoma": 1}).to_numpy()
groups = labels["patient_id"].to_numpy()

# Confirm every patient has exactly one sample of each tissue.
for patient, indices in labels.groupby("patient_id").groups.items():
    patient_labels = labels.loc[list(indices), "group"].tolist()
    if sorted(patient_labels) != ["Adenoma", "Normal"]:
        raise ValueError(
            f"Patient {patient} does not have one sample of each tissue: "
            f"{patient_labels}"
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

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def evaluate(labels_array):
    pred = cross_val_predict(
        clone(pipeline),
        X,
        labels_array,
        groups=groups,
        cv=cv,
        method="predict"
    )
    return accuracy_score(labels_array, pred)

observed = evaluate(y)
print("Observed grouped-CV accuracy:", observed)

# Swap Normal/Adenoma labels independently within each patient's pair.
rng = np.random.default_rng(42)
permutation_scores = []
n_permutations = 100

for i in range(n_permutations):
    y_perm = y.copy()

    for patient in np.unique(groups):
        idx = np.flatnonzero(groups == patient)
        if rng.random() < 0.5:
            y_perm[idx] = y_perm[idx][::-1]

    # Feature selection is refitted inside each fold by the pipeline.
    score = evaluate(y_perm)
    permutation_scores.append(score)

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{n_permutations}")

scores = np.asarray(permutation_scores)

# Plus-one correction for a Monte Carlo permutation test.
p_value = (
    1 + np.count_nonzero(scores >= observed)
) / (n_permutations + 1)

print("\nPermutation results")
print("Mean accuracy:", round(float(scores.mean()), 3))
print("Maximum accuracy:", round(float(scores.max()), 3))
print("Observed accuracy:", round(float(observed), 3))
print("Approximate permutation p-value:", round(float(p_value), 4))

out = ROOT / "results" / "paired_permutation_scores.csv"
pd.DataFrame({"accuracy": scores}).to_csv(out, index=False)
print("Saved permutation scores:", out)

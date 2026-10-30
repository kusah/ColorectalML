
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "gene_level"

X = pd.read_csv(DATA / "X_train.csv", index_col="sample_id")
y = pd.read_csv(DATA / "y_train.csv", index_col="sample_id")["label"]
meta = pd.read_csv(
    ROOT / "data" / "processed" / "sample_labels.csv",
    dtype={"sample_id": str}
).drop_duplicates("sample_id").set_index("sample_id")

X.index = X.index.astype(str)
y.index = y.index.astype(str)
meta.index = meta.index.astype(str)

y = y.loc[X.index].astype(int)
groups = meta.loc[X.index, "patient_id"]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("select", SelectKBest(f_classif, k=min(100, X.shape[1]))),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )),
])


def rank_transform_samples(X):
    arr = X.astype(float).to_numpy()
    result = np.full(arr.shape, np.nan, dtype=float)

    for i in range(arr.shape[0]):
        valid = np.isfinite(arr[i])
        if valid.sum() > 1:
            result[i, valid] = (
                rankdata(arr[i, valid]) - 1
            ) / (valid.sum() - 1)
        elif valid.sum() == 1:
            result[i, valid] = 0.5

    return pd.DataFrame(result, index=X.index, columns=X.columns)
X = rank_transform_samples(X)
cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

prob = cross_val_predict(
    pipeline, X, y, groups=groups, cv=cv,
    method="predict_proba", n_jobs=-1
)[:, 1]

pred = (prob >= 0.5).astype(int)

print("Training samples:", len(X))
print("Unique patients:", groups.nunique())
print("Labels aligned:", X.index.equals(y.index))
print("Missing feature values:", int(X.isna().sum().sum()))

print("\nCross-validated probability summary:")
print(pd.DataFrame({
    "true_label": y.to_numpy(),
    "adenoma_probability": prob
}).groupby("true_label")["adenoma_probability"].describe())

print("\nConfusion matrix (Normal, Adenoma):")
print(confusion_matrix(y, pred, labels=[0, 1]))
print("Cross-validated ROC-AUC:", roc_auc_score(y, prob))
print("\nFold-level patient overlap checks:")
for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
    overlap = set(groups.iloc[tr]) & set(groups.iloc[te])
    print(
        f"Fold {fold}: train={len(tr)}, test={len(te)}, "
        f"patient overlap={len(overlap)}"
    )

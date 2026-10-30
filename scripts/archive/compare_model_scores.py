
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "gene_level"
MODEL = ROOT / "models" / "gene_level_logistic_model.joblib"

def rank_transform_samples(X):
    import numpy as np
    arr = X.astype(float).to_numpy()
    result = np.full(arr.shape, np.nan)

    for i in range(arr.shape[0]):
        valid = np.isfinite(arr[i])
        if valid.sum() > 1:
            result[i, valid] = (
                rankdata(arr[i, valid]) - 1
            ) / (valid.sum() - 1)
        elif valid.sum() == 1:
            result[i, valid] = 0.5

    return pd.DataFrame(result, index=X.index, columns=X.columns)

X_train = pd.read_csv(DATA / "X_train.csv", index_col="sample_id")
X_external = pd.read_csv(DATA / "X_external.csv", index_col="sample_id")
y_train = pd.read_csv(DATA / "y_train.csv", index_col="sample_id")["label"]

X_train = rank_transform_samples(X_train)
X_external = rank_transform_samples(X_external)

model = joblib.load(MODEL)

# Decision scores are useful for comparing ranking and separation.
train_scores = model.decision_function(X_train)
external_scores = model.decision_function(X_external)

train_labels = y_train.loc[X_train.index].map({
    0: "Normal", 1: "Adenoma"
})

external_meta = pd.read_csv(
    ROOT / "data" / "processed" / "GSE50114" / "sample_metadata.csv",
    dtype={"sample_id": str}
).set_index("sample_id")
external_labels = external_meta.loc[X_external.index, "tissue"]

print("TRAINING DECISION SCORES")
print(pd.DataFrame({
    "group": train_labels.to_numpy(),
    "score": train_scores
}).groupby("group")["score"].describe())

print("\nEXTERNAL DECISION SCORES")
print(pd.DataFrame({
    "group": external_labels.to_numpy(),
    "score": external_scores
}).groupby("group")["score"].describe())

print("\nExternal scores by sample:")
print(pd.DataFrame({
    "group": external_labels,
    "score": external_scores
}).sort_values("score").to_string())

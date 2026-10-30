
from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "gene_level"
META_DIR = ROOT / "data" / "processed" / "GSE50114"
OUT_DIR = ROOT / "results" / "external_validation"
MODEL_DIR = ROOT / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TOP_K = 100


def rank_transform_samples(X):
    """Transform each sample independently to ranks scaled from 0 to 1."""
    X = X.astype(float).copy()
    arr = X.to_numpy()

    transformed = np.full(arr.shape, np.nan, dtype=float)

    for i in range(arr.shape[0]):
        valid = np.isfinite(arr[i])
        if valid.sum() > 1:
            transformed[i, valid] = (
                rankdata(arr[i, valid], method="average") - 1
            ) / (valid.sum() - 1)
        elif valid.sum() == 1:
            transformed[i, valid] = 0.5

    return pd.DataFrame(
        transformed, index=X.index, columns=X.columns
    )


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_adenoma": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "sensitivity_adenoma": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_adenoma": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2 else np.nan
        ),
    }


# -------------------------------------------------------
# 1. Load gene-level matrices and training labels
# -------------------------------------------------------
X_train = pd.read_csv(
    DATA_DIR / "X_train.csv", index_col="sample_id"
)
y_df = pd.read_csv(
    DATA_DIR / "y_train.csv", index_col="sample_id"
)
X_external = pd.read_csv(
    DATA_DIR / "X_external.csv", index_col="sample_id"
)

X_train.index = X_train.index.astype(str)
X_external.index = X_external.index.astype(str)
y_df.index = y_df.index.astype(str)

y_train = y_df.loc[X_train.index, "label"].astype(int)

# Patient IDs are required for grouped cross-validation.
train_labels = pd.read_csv(
    ROOT / "data" / "processed" / "sample_labels.csv",
    dtype={"sample_id": str}
).drop_duplicates("sample_id").set_index("sample_id")

groups = train_labels.loc[X_train.index, "patient_id"]

if groups.isna().any():
    raise ValueError("Missing patient IDs in training metadata.")

if not X_train.columns.equals(X_external.columns):
    raise ValueError("Training and external gene columns differ.")

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training labels must contain 0=Normal and 1=Adenoma.")

print("Training shape:", X_train.shape)
print("External shape:", X_external.shape)
print("Training labels:")
print(y_train.value_counts().sort_index())
print("Unique training patients:", groups.nunique())

# -------------------------------------------------------
# 2. Load and validate external labels
# -------------------------------------------------------
external_meta = pd.read_csv(
    META_DIR / "sample_metadata.csv",
    dtype={"sample_id": str}
).drop_duplicates("sample_id").set_index("sample_id")

external_meta = external_meta.reindex(X_external.index)

if external_meta["tissue"].isna().any():
    missing = external_meta.index[
        external_meta["tissue"].isna()
    ].tolist()
    raise ValueError(f"Missing external metadata for: {missing}")

label_map = {"Normal": 0, "Adenoma": 1}
y_external = external_meta["tissue"].map(label_map)

if y_external.isna().any():
    raise ValueError(
        "Unexpected external labels: "
        + str(external_meta["tissue"].unique())
    )

y_external = y_external.astype(int)

print("\nExternal label counts:")
print(y_external.value_counts().sort_index())

# -------------------------------------------------------
# 3. Rank-transform each sample independently
# -------------------------------------------------------
# This uses no class labels. It is applied identically to
# training and external samples.
X_train_ranked = X_train
X_external_ranked = X_external

# -------------------------------------------------------
# 4. Model pipeline
# All learned preprocessing and feature selection occur
# within each training fold.
# -------------------------------------------------------
k = min(TOP_K, X_train.shape[1])

pipeline = Pipeline([
    (
        "rank",
        FunctionTransformer(
            rank_transform_samples,
            validate=False
        )
    ),
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "select",
        SelectKBest(
            score_func=f_classif,
            k=k
        )
    ),
    (
        "scale",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    ),
])

# -------------------------------------------------------
# 5. Patient-grouped cross-validation on GSE8671
# -------------------------------------------------------
cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


oof_probabilities = cross_val_predict(
    pipeline,
    X_train,
    y_train,
    groups=groups,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

oof_prob = oof_probabilities[:, 1]
oof_pred = (oof_probabilities.argmax(axis=1)).astype(int)


cv_metrics = calculate_metrics(y_train, oof_pred, oof_prob)

print("\n========== GROUPED CV RESULTS ==========")
for metric, value in cv_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nCross-validation classification report:")
print(classification_report(
    y_train,
    oof_pred,
    labels=[0, 1],
    target_names=["Normal", "Adenoma"],
    zero_division=0
))

# Verify patient separation in every fold.
for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_train_ranked, y_train, groups), start=1
):
    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])

    if train_patients & test_patients:
        raise ValueError(f"Patient leakage detected in fold {fold}")

# -------------------------------------------------------
# 6. Fit final model on all GSE8671 samples
# -------------------------------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------------------------------
# 7. External validation on GSE50114
# -------------------------------------------------------
external_pred = pipeline.predict(X_external)
external_prob = pipeline.predict_proba(X_external)[:, 1]

external_metrics = calculate_metrics(
    y_external, external_pred, external_prob
)

print("\n========== EXTERNAL VALIDATION ==========")
for metric, value in external_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nExternal classification report:")
print(classification_report(
    y_external,
    external_pred,
    labels=[0, 1],
    target_names=["Normal", "Adenoma"],
    zero_division=0
))

# -------------------------------------------------------
# 8. Save results and model
# -------------------------------------------------------
results = pd.DataFrame([
    {"evaluation": "grouped_cv", **cv_metrics},
    {"evaluation": "external_GSE50114", **external_metrics},
])
results.to_csv(OUT_DIR / "metrics.csv", index=False)

predictions = pd.DataFrame({
    "sample_id": X_external.index,
    "true_label": y_external.to_numpy(),
    "predicted_label": external_pred,
    "adenoma_probability": external_prob,
    "true_group": y_external.map({
        0: "Normal", 1: "Adenoma"
    }).to_numpy(),
    "predicted_group": pd.Series(external_pred).map({
        0: "Normal", 1: "Adenoma"
    }).to_numpy(),
})
predictions.to_csv(
    OUT_DIR / "external_predictions.csv", index=False
)

cm = confusion_matrix(y_external, external_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(cm, cmap="Blues")
ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Normal", "Adenoma"],
    yticklabels=["Normal", "Adenoma"],
    xlabel="Predicted label",
    ylabel="True label",
    title="GSE50114 External Validation"
)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(
    OUT_DIR / "external_confusion_matrix.png", dpi=200
)
plt.close(fig)

joblib.dump(
    pipeline, MODEL_DIR / "gene_level_logistic_model.joblib"
)

print("\nSaved:")
print(OUT_DIR / "metrics.csv")
print(OUT_DIR / "external_predictions.csv")
print(OUT_DIR / "external_confusion_matrix.png")
print(MODEL_DIR / "gene_level_logistic_model.joblib")
print("\nResearch prototype only; not a clinical diagnostic tool.")

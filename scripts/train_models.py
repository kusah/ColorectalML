
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT / "models"

for folder in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 1. Load prepared data
X_df = pd.read_csv(
    DATA_DIR / "X_expression.csv",
    index_col=0
)

labels = pd.read_csv(DATA_DIR / "sample_labels.csv")
labels = labels.set_index("sample_id").loc[X_df.index]

y = labels["group"].map({
    "Normal": 0,
    "Adenoma": 1
}).to_numpy()

groups = labels["patient_id"].to_numpy()

if pd.isna(y).any():
    raise ValueError("Unexpected group labels found.")

if len(np.unique(y)) != 2:
    raise ValueError("Expected Normal and Adenoma classes.")

# Each patient must have samples from only one CV group.
# Samples belonging to the same patient stay in the same fold.
X = X_df.to_numpy(dtype=np.float64)

print("Dataset shape:", X.shape)
print("Normal samples:", int((y == 0).sum()))
print("Adenoma samples:", int((y == 1).sum()))
print("Unique patient IDs:", len(np.unique(groups)))

# 2. Group-aware stratified cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 3. Models
# Feature selection and scaling are fitted separately inside each fold.
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=100)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=100)),
        ("scale", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=100)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# 4. Evaluate using out-of-fold predictions
all_metrics = []
saved_models = {}

for name, model in models.items():
    print(f"\nTraining/evaluating: {name}")

    # Every prediction is made for a patient group held out from training.
    predictions = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict",
        n_jobs=1
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y, probabilities)
    }
    all_metrics.append(metrics)

    print(pd.Series(metrics).to_string())
    print("\nClassification report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Adenoma"],
        zero_division=0
    ))

    # Save out-of-fold predictions
    prediction_df = pd.DataFrame({
        "sample_id": X_df.index,
        "patient_id": groups,
        "actual": np.where(y == 1, "Adenoma", "Normal"),
        "predicted": np.where(
            predictions == 1, "Adenoma", "Normal"
        ),
        "adenoma_probability": probabilities
    })
    safe_name = name.lower().replace(" ", "_")
    prediction_df.to_csv(
        RESULTS_DIR / f"{safe_name}_predictions.csv",
        index=False
    )

    # Save confusion matrix
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Adenoma"]
    )
    display.plot(values_format="d")
    plt.title(f"{name} — Grouped CV")
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=200
    )
    plt.close()

    # Fit a final model on all samples for the demo.
    # This saved model is for demonstration, not independent validation.
    model.fit(X, y)
    saved_models[name] = model

    joblib.dump(
        model,
        MODELS_DIR / f"{safe_name}.joblib"
    )

# 5. Save comparison table
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.sort_values(
    by="ROC AUC",
    ascending=False
)
metrics_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

with open(RESULTS_DIR / "model_comparison.json", "w") as f:
    json.dump(
        metrics_df.to_dict(orient="records"),
        f,
        indent=2
    )

print("\nMODEL COMPARISON")
print(metrics_df.to_string(index=False))
print("\nSaved metrics:", RESULTS_DIR / "model_comparison.csv")
print("Saved plots:", PLOTS_DIR)
print("Saved demo models:", MODELS_DIR)
print("\nTraining and evaluation completed.")

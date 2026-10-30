
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "external_validation"

metrics = pd.read_csv(RESULTS / "metrics.csv")
pred = pd.read_csv(RESULTS / "external_predictions.csv")

print("========== SAVED METRICS ==========")
print(metrics.to_string(index=False))

print("\n========== PREDICTION DATA ==========")
print("Shape:", pred.shape)
print("Columns:", pred.columns.tolist())
print("\nTrue labels:")
print(pred["true_group"].value_counts(dropna=False))

print("\nPredicted labels:")
print(pred["predicted_group"].value_counts(dropna=False))

print("\n========== PROBABILITY DISTRIBUTION ==========")
print(pred.groupby("true_group")["adenoma_probability"].describe())

print("\n========== PREDICTION CHECKS ==========")
print("Missing probabilities:",
      pred["adenoma_probability"].isna().sum())
print("Probabilities outside [0, 1]:",
      ((pred["adenoma_probability"] < 0) |
       (pred["adenoma_probability"] > 1)).sum())

print("\nAdenoma probability range:")
print(pred["adenoma_probability"].agg(["min", "median", "max"]))

y_true = pred["true_label"]
y_pred = pred["predicted_label"]
y_prob = pred["adenoma_probability"]

print("\nConfusion matrix (Normal, Adenoma):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

if y_true.nunique() == 2:
    print("\nRecomputed ROC-AUC:",
          roc_auc_score(y_true, y_prob))

print("\n========== SAMPLE PREDICTIONS ==========")
print(
    pred.sort_values("adenoma_probability")[
        ["sample_id", "true_group", "predicted_group",
         "adenoma_probability"]
    ].to_string(index=False)
)

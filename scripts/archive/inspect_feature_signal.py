
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import f_classif

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

X = pd.read_csv(DATA_DIR / "X_expression.csv", index_col=0)
labels = pd.read_csv(DATA_DIR / "sample_labels.csv").set_index("sample_id")
labels = labels.loc[X.index]

normal_ids = labels.index[labels["group"] == "Normal"]
adenoma_ids = labels.index[labels["group"] == "Adenoma"]

normal = X.loc[normal_ids]
adenoma = X.loc[adenoma_ids]

print("Normal shape:", normal.shape)
print("Adenoma shape:", adenoma.shape)

# Compare overall expression distributions
print("\n=== Expression distribution ===")
print("Normal median:", float(np.nanmedian(normal.to_numpy())))
print("Adenoma median:", float(np.nanmedian(adenoma.to_numpy())))
print("Normal standard deviation:", float(np.nanstd(normal.to_numpy())))
print("Adenoma standard deviation:", float(np.nanstd(adenoma.to_numpy())))

# Rank probes using the full dataset for diagnosis only.
# This is exploratory and must NOT be used as an unbiased performance estimate.
f_scores, p_values = f_classif(
    X.to_numpy(),
    labels["group"].map({"Normal": 0, "Adenoma": 1}).to_numpy()
)

results = pd.DataFrame({
    "probe_id": X.columns,
    "f_score": f_scores,
    "p_value": p_values,
    "normal_mean": normal.mean(axis=0).to_numpy(),
    "adenoma_mean": adenoma.mean(axis=0).to_numpy()
})

results["mean_difference"] = (
    results["adenoma_mean"] - results["normal_mean"]
)

results = results.replace([np.inf, -np.inf], np.nan).dropna()
results = results.sort_values("f_score", ascending=False)

output = RESULTS_DIR / "exploratory_top_probes.csv"
results.head(100).to_csv(output, index=False)

print("\nTop 20 probes by F-score:")
print(results.head(20).to_string(index=False))
print("\nSaved:", output)

# Simple sanity check: how different are the sample-level mean profiles?
sample_means = X.mean(axis=1).to_frame("mean_expression")
sample_means["group"] = labels["group"]

print("\nMean expression by class:")
print(sample_means.groupby("group")["mean_expression"].describe())

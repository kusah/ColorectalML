
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "gene_level"
MODEL_PATH = ROOT / "models" / "gene_level_logistic_model.joblib"

model = joblib.load(MODEL_PATH)

X_train = pd.read_csv(DATA / "X_train.csv", index_col="sample_id")
X_external = pd.read_csv(DATA / "X_external.csv", index_col="sample_id")

def rank_transform(X):
    from scipy.stats import rankdata

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

X_train = rank_transform(X_train)
X_external = rank_transform(X_external)

# The model pipeline's feature selector is fitted on training data.
selector = model.named_steps["select"]
selected_mask = selector.get_support()
selected_genes = X_train.columns[selected_mask]

print("Total shared genes:", X_train.shape[1])
print("Selected genes:", len(selected_genes))
print("Selected genes unique:", selected_genes.is_unique)
print("Feature columns aligned:", X_train.columns.equals(X_external.columns))

# Compare selected-gene distributions between studies.
train_selected = X_train.loc[:, selected_genes]
external_selected = X_external.loc[:, selected_genes]

summary = pd.DataFrame({
    "train_mean": train_selected.mean(),
    "external_mean": external_selected.mean(),
    "train_std": train_selected.std(),
    "external_std": external_selected.std(),
})

summary["absolute_mean_difference"] = (
    summary["train_mean"] - summary["external_mean"]
).abs()

summary = summary.sort_values(
    "absolute_mean_difference", ascending=False
)

print("\nLargest selected-gene distribution differences:")
print(summary.head(20).to_string())

print("\nCoefficient magnitude:")
coefficients = model.named_steps["model"].coef_[0]
coef_table = pd.DataFrame({
    "gene": selected_genes,
    "coefficient": coefficients,
})
coef_table["absolute_coefficient"] = coef_table["coefficient"].abs()

print(
    coef_table.sort_values(
        "absolute_coefficient", ascending=False
    ).head(20).to_string(index=False)
)

summary.to_csv(ROOT / "results" / "external_validation" /
               "selected_gene_distribution_audit.csv")
coef_table.to_csv(ROOT / "results" / "external_validation" /
                  "selected_gene_coefficients.csv", index=False)

print("\nAudit files saved in results/external_validation/")

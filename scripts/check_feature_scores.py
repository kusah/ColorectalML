from pathlib import Path
import pandas as pd
from sklearn.feature_selection import f_classif

root = Path(__file__).resolve().parent.parent

X = pd.read_csv(root / "data/processed/X_expression.csv", index_col=0)
labels = pd.read_csv(root / "data/processed/sample_labels.csv")
labels = labels.set_index("sample_id").loc[X.index]

y = labels["group"].map({"Normal": 0, "Adenoma": 1}).to_numpy()

scores, pvalues = f_classif(X.to_numpy(dtype=float), y)

results = pd.DataFrame({
    "Gene": X.columns,
    "ANOVA_F_score": scores,
    "p_value": pvalues
}).sort_values("ANOVA_F_score", ascending=False)

results.head(100).to_csv(
    root / "results/top_100_feature_scores.csv",
    index=False
)

print(results.head(10).to_string(index=False))
print("\nSaved top 100 feature scores.")
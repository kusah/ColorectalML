
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "common_genes"

gse8671 = pd.read_csv(
    DATA_DIR / "GSE8671_common_genes.csv", index_col=0
)
gse50114 = pd.read_csv(
    DATA_DIR / "GSE50114_common_genes.csv", index_col=0
)

# Missingness by sample and by gene
sample_missing = gse50114.isna().sum(axis=0)
gene_missing = gse50114.isna().sum(axis=1)

print("External dataset shape:", gse50114.shape)
print("Total missing values:", int(gse50114.isna().sum().sum()))
print(
    "Overall missing percentage:",
    round(gse50114.isna().mean().mean() * 100, 4), "%"
)

print("\nMissing values by sample:")
print(sample_missing[sample_missing > 0].sort_values(ascending=False))

print("\nMissing values by gene:")
print(gene_missing[gene_missing > 0].sort_values(ascending=False).head(20))

print("\nSamples with any missing values:", int((sample_missing > 0).sum()))
print("Genes with any missing values:", int((gene_missing > 0).sum()))

# Check whether any training genes have zero variance
train_std = gse8671.std(axis=1)
print("\nZero-variance genes in GSE8671:", int((train_std == 0).sum()))

# Confirm all values that are present are numeric
print("\nExternal dataset numeric conversion check:")
numeric = gse50114.apply(pd.to_numeric, errors="coerce")
new_na = numeric.isna().sum().sum() - gse50114.isna().sum().sum()
print("Additional non-numeric values converted to missing:", int(new_na))

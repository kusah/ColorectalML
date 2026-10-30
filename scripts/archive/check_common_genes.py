
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "common_genes"

files = {
    "GSE8671": DATA_DIR / "GSE8671_common_genes.csv",
    "GSE50114": DATA_DIR / "GSE50114_common_genes.csv",
    "shared_genes": DATA_DIR / "shared_gene_symbols.csv",
}

for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

gse8671 = pd.read_csv(files["GSE8671"], index_col=0)
gse50114 = pd.read_csv(files["GSE50114"], index_col=0)
shared = pd.read_csv(files["shared_genes"])["gene_symbol"]

print("GSE8671 aligned shape:", gse8671.shape)
print("GSE50114 aligned shape:", gse50114.shape)
print("Shared gene count:", len(shared))

print("\nGene indexes identical:",
      gse8671.index.equals(gse50114.index))

print("Genes unique in shared list:",
      shared.is_unique)

print("Missing values in GSE8671:", gse8671.isna().sum().sum())
print("Missing values in GSE50114:", gse50114.isna().sum().sum())

print("\nGSE8671 sample count:", gse8671.shape[1])
print("GSE50114 sample count:", gse50114.shape[1])

assert gse8671.index.equals(gse50114.index), (
    "Gene order differs between datasets."
)
assert shared.is_unique, "Duplicate genes found in shared list."

print("\nAlignment checks completed.")

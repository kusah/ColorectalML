
from pathlib import Path
import re

import GEOparse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Input files
GSE8671_SOFT = ROOT / "data" / "raw" / "GSE8671_family.soft.gz"
GSE50114_DIR = ROOT / "data" / "raw" / "GSE50114"

GSE8671_EXPR = ROOT / "data" / "processed" / "expression_matrix.csv"
GSE50114_EXPR = (
    ROOT / "data" / "processed" / "GSE50114" / "expression_matrix.csv"
)

OUT_DIR = ROOT / "data" / "processed" / "common_genes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_soft_file(path):
    """Find a local GEO SOFT file."""
    if path.is_file():
        return path

    files = list(path.glob("*.soft.gz"))
    if not files:
        raise FileNotFoundError(f"No SOFT file found in: {path}")

    return files[0]


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_symbol_column(platform_table):
    """Find a likely gene-symbol column in a GEO platform table."""
    normalized = {
        normalize_column_name(col): col
        for col in platform_table.columns
    }

    candidates = [
        "gene_symbol",
        "genesymbol",
        "gene symbols",
        "genesymbols",
        "symbol",
    ]

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    print("Available platform columns:")
    print(platform_table.columns.tolist())

    raise ValueError(
        "Could not identify a gene-symbol column. "
        "Inspect the printed platform columns before proceeding."
    )


def load_platform_annotation(soft_path, dataset_name):
    """Load the platform probe-to-gene annotation from a local SOFT file."""
    print(f"\nLoading platform annotation: {dataset_name}")

    gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)

    if not gse.gpls:
        raise ValueError(
            f"No platform table found in {soft_path}. "
            "We may need to download the platform annotation separately."
        )

    annotations = []

    for platform_id, platform in gse.gpls.items():
        table = platform.table.copy()

        if table.empty or "ID" not in table.columns:
            continue

        symbol_col = find_symbol_column(table)

        part = table[["ID", symbol_col]].copy()
        part.columns = ["probe_id", "gene_symbol"]
        part["platform"] = platform_id

        annotations.append(part)

        print(f"Platform: {platform_id}")
        print(f"Probe annotation rows: {len(part):,}")
        print(f"Gene symbol column: {symbol_col}")

    if not annotations:
        raise ValueError(
            f"No usable platform annotation found for {dataset_name}."
        )

    return pd.concat(annotations, ignore_index=True)


def clean_single_gene_symbol(value):
    """
    Keep only unambiguous single-gene annotations.
    Multi-gene assignments are excluded to avoid incorrect mapping.
    """
    if pd.isna(value):
        return None

    symbol = str(value).strip()

    if not symbol or symbol.lower() in {
        "na", "nan", "null", "---", "none"
    }:
        return None

    # Common separators for multiple gene assignments
    if any(sep in symbol for sep in ["///", "//", ";", "|"]):
        return None

    if symbol in {"-", "---"}:
        return None

    return symbol.upper()


def map_probes_to_genes(expression_path, annotation, dataset_name):
    """Map probe rows to gene symbols and aggregate probes per gene."""
    expression = pd.read_csv(expression_path)

    # Expression files should have probe_id in the first column.
    if "probe_id" not in expression.columns:
        expression = expression.rename(
            columns={expression.columns[0]: "probe_id"}
        )

    expression["probe_id"] = expression["probe_id"].astype(str)

    annotation = annotation.copy()
    annotation["probe_id"] = annotation["probe_id"].astype(str)
    annotation["gene_symbol"] = annotation["gene_symbol"].apply(
        clean_single_gene_symbol
    )

    annotation = annotation.dropna(subset=["gene_symbol"])
    annotation = annotation.drop_duplicates(subset=["probe_id"])

    merged = expression.merge(
        annotation[["probe_id", "gene_symbol"]],
        on="probe_id",
        how="inner",
        validate="many_to_one",
    )

    sample_columns = [
        col for col in expression.columns if col != "probe_id"
    ]

    for col in sample_columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    mapped_probe_count = merged["probe_id"].nunique()
    total_probe_count = expression["probe_id"].nunique()

    # Multiple probes may measure the same gene.
    # Median aggregation reduces the influence of extreme probe values.
    gene_expression = merged.groupby("gene_symbol")[
        sample_columns
    ].median()

    gene_expression.index.name = "gene_symbol"

    print(f"\n{dataset_name} mapping summary")
    print(f"Input probes: {total_probe_count:,}")
    print(f"Mapped probes: {mapped_probe_count:,}")
    print(f"Mapped genes: {gene_expression.shape[0]:,}")
    print(f"Samples: {gene_expression.shape[1]:,}")

    gene_expression.to_csv(
        OUT_DIR / f"{dataset_name}_gene_expression.csv"
    )

    return gene_expression


def main():
    soft_8671 = find_soft_file(GSE8671_SOFT)
    soft_50114 = find_soft_file(GSE50114_DIR)

    annotation_8671 = load_platform_annotation(
        soft_8671, "GSE8671"
    )
    annotation_50114 = load_platform_annotation(
        soft_50114, "GSE50114"
    )

    genes_8671 = map_probes_to_genes(
        GSE8671_EXPR, annotation_8671, "GSE8671"
    )
    genes_50114 = map_probes_to_genes(
        GSE50114_EXPR, annotation_50114, "GSE50114"
    )

    common_genes = sorted(
        set(genes_8671.index) & set(genes_50114.index)
    )

    if not common_genes:
        raise ValueError(
            "No shared gene symbols found. Check platform annotations."
        )

    aligned_8671 = genes_8671.loc[common_genes].sort_index()
    aligned_50114 = genes_50114.loc[common_genes].sort_index()

    aligned_8671.to_csv(
        OUT_DIR / "GSE8671_common_genes.csv"
    )
    aligned_50114.to_csv(
        OUT_DIR / "GSE50114_common_genes.csv"
    )

    pd.Series(
        common_genes, name="gene_symbol"
    ).to_csv(
        OUT_DIR / "shared_gene_symbols.csv",
        index=False,
    )

    print("\n========== FINAL SUMMARY ==========")
    print(f"Genes in GSE8671: {len(genes_8671):,}")
    print(f"Genes in GSE50114: {len(genes_50114):,}")
    print(f"Shared genes: {len(common_genes):,}")
    print("\nSaved files in:", OUT_DIR)
    print("  GSE8671_gene_expression.csv")
    print("  GSE50114_gene_expression.csv")
    print("  GSE8671_common_genes.csv")
    print("  GSE50114_common_genes.csv")
    print("  shared_gene_symbols.csv")


if __name__ == "__main__":
    main()

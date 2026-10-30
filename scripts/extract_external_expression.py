
from pathlib import Path
import GEOparse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "GSE50114"
OUT_DIR = ROOT / "data" / "processed" / "GSE50114"
OUT_DIR.mkdir(parents=True, exist_ok=True)

soft_files = list(DATA_DIR.glob("*.soft.gz"))
if not soft_files:
    raise FileNotFoundError(f"No .soft.gz file found in {DATA_DIR}")

gse = GEOparse.get_GEO(filepath=str(soft_files[0]))

expression_columns = {}
metadata_rows = []

for gsm_id, gsm in gse.gsms.items():
    table = gsm.table

    if "ID_REF" not in table.columns or "VALUE" not in table.columns:
        print(f"Skipping {gsm_id}: ID_REF/VALUE not available")
        continue

    values = table[["ID_REF", "VALUE"]].copy()
    values["VALUE"] = pd.to_numeric(values["VALUE"], errors="coerce")
    values = values.dropna(subset=["ID_REF", "VALUE"])
    values = values.drop_duplicates(subset=["ID_REF"], keep="first")

    expression_columns[gsm_id] = values.set_index("ID_REF")["VALUE"]

    title = gsm.metadata.get("title", [""])[0]
    source = gsm.metadata.get("source_name_ch1", [""])[0]
    characteristics = gsm.metadata.get("characteristics_ch1", [])

    tissue = "Unknown"
    for item in characteristics:
        low = item.lower()
        if "tissue:" in low:
            if "adenoma" in low:
                tissue = "Adenoma"
            elif "normal" in low:
                tissue = "Normal"

    metadata_rows.append({
        "sample_id": gsm_id,
        "title": title,
        "source": source,
        "tissue": tissue,
        "characteristics": " | ".join(characteristics)
    })

if not expression_columns:
    raise ValueError(
        "No sample-level ID_REF/VALUE tables found. "
        "We need to inspect the GEO supplementary files instead."
    )

expression = pd.DataFrame(expression_columns)
metadata = pd.DataFrame(metadata_rows)

expression.index.name = "probe_id"
expression.to_csv(OUT_DIR / "expression_matrix.csv")
metadata.to_csv(OUT_DIR / "sample_metadata.csv", index=False)

print("Expression matrix shape:", expression.shape)
print("Metadata shape:", metadata.shape)
print("\nTissue counts:")
print(metadata["tissue"].value_counts(dropna=False))
print("\nExpression preview:")
print(expression.iloc[:5, :5])
print("\nSaved files to:", OUT_DIR)


from pathlib import Path
import pandas as pd
import GEOparse

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

soft_file = RAW_DIR / "GSE8671_family.soft.gz"

if not soft_file.exists():
    raise FileNotFoundError(f"Missing GEO file: {soft_file}")

print("Loading GSE8671 from the downloaded file...")
gse = GEOparse.get_GEO(filepath=str(soft_file))

print("\nSeries:", gse.name)
print("Number of samples:", len(gse.gsms))

metadata_rows = []
expression_columns = []

for gsm_id, gsm in gse.gsms.items():
    title = str(gsm.metadata.get("title", [""])[0])
    source = str(
        gsm.metadata.get("source_name_ch1", [""])[0]
    )
    characteristics = gsm.metadata.get(
        "characteristics_ch1", []
    )

    metadata_rows.append({
        "sample_id": gsm_id,
        "title": title,
        "source": source,
        "characteristics": " | ".join(characteristics)
    })

    table = gsm.table.copy()

    print(f"\nSample: {gsm_id} | {title}")
    print("Table columns:", list(table.columns))
    print("Table shape:", table.shape)

    if {"ID_REF", "VALUE"}.issubset(table.columns):
        values = table[["ID_REF", "VALUE"]].copy()
        values["VALUE"] = pd.to_numeric(
            values["VALUE"], errors="coerce"
        )
        values = values.dropna(subset=["VALUE"])
        values = values.drop_duplicates(
            subset="ID_REF", keep="first"
        )
        values = values.set_index("ID_REF")

        expression_columns.append(
            values["VALUE"].rename(gsm_id)
        )

metadata = pd.DataFrame(metadata_rows)

metadata_path = OUT_DIR / "sample_metadata.csv"
metadata.to_csv(metadata_path, index=False)

print("\n--- Sample metadata preview ---")
print(metadata.head(10).to_string(index=False))

print("\n--- Sample titles and sources ---")
for _, row in metadata.iterrows():
    print(
        f"{row['sample_id']}: "
        f"title={row['title']!r}; "
        f"source={row['source']!r}; "
        f"characteristics={row['characteristics']!r}"
    )

if expression_columns:
    expression = pd.concat(
        expression_columns, axis=1, join="outer"
    )
    expression.index.name = "probe_id"

    expression_path = OUT_DIR / "expression_matrix.csv"
    expression.to_csv(expression_path)

    print("\nExpression matrix shape:", expression.shape)
    print("Expression matrix saved:", expression_path)
else:
    print(
        "\nNo processed ID_REF/VALUE tables were found "
        "in the downloaded SOFT file."
    )
    print(
        "We will obtain the expression matrix separately "
        "from the GEO supplementary or series-matrix files."
    )

print("\nMetadata saved:", metadata_path)
print("Inspection complete.")

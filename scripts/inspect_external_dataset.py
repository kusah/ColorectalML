
from pathlib import Path
import GEOparse

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "GSE50114"

print("Looking for downloaded GEO files...")

soft_files = list(DATA_DIR.glob("*.soft.gz"))

if soft_files:
    gse = GEOparse.get_GEO(filepath=str(soft_files[0]))
else:
    gse = GEOparse.get_GEO(
        "GSE50114",
        destdir=str(DATA_DIR),
        silent=False
    )

print("\n=== Dataset summary ===")
print("Dataset ID: GSE50114")
print("Title:", gse.metadata.get("title", ["Unknown"])[0])
print("Platforms:", list(gse.gpls.keys()))
print("Total samples:", len(gse.gsms))

print("\n=== Sample metadata ===")
for gsm_id, gsm in gse.gsms.items():
    print("\nSample:", gsm_id)
    print("Title:", gsm.metadata.get("title", ["Unknown"])[0])
    print("Source:", gsm.metadata.get("source_name_ch1", ["Unknown"])[0])
    print("Characteristics:")
    for item in gsm.metadata.get("characteristics_ch1", []):
        print("  ", item)

print("\n=== Platform information ===")
for gpl_id, gpl in gse.gpls.items():
    print("Platform:", gpl_id)
    print("Annotation columns:", list(gpl.table.columns))
    print("Annotation preview:")
    print(gpl.table.head(3).to_string(index=False))

print("\nInspection completed.")

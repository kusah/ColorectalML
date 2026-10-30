
from pathlib import Path
import GEOparse

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "GSE50114"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading GSE50114 from NCBI GEO...")

gse = GEOparse.get_GEO(
    "GSE50114",
    destdir=str(RAW_DIR),
    silent=False
)

print("\nDownload and parsing completed.")
print("Dataset:", gse.name)
print("Title:", gse.metadata.get("title", ["Unknown"])[0])
print("Platform IDs:", list(gse.gpls.keys()))
print("Number of samples:", len(gse.gsms))

print("\nSample overview:")
for gsm_id, gsm in list(gse.gsms.items())[:10]:
    print(
        gsm_id,
        "|",
        gsm.metadata.get("title", ["Unknown"])[0]
    )

print("\nSaved GEO files to:", RAW_DIR)

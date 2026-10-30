
from pathlib import Path
import gzip
import time
import requests
import GEOparse

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

url = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE8nnn/GSE8671/soft/GSE8671_family.soft.gz"
)

archive = RAW_DIR / "GSE8671_family.soft.gz"
session = requests.Session()
session.headers.update({
    "User-Agent": "ColorectalCancerML/1.0 (academic project)"
})

# Download with retries. Stream the response to disk.
for attempt in range(1, 4):
    try:
        print(f"Download attempt {attempt}/3")

        with session.get(
            url, stream=True, timeout=(30, 180)
        ) as response:
            response.raise_for_status()
            temp = archive.with_suffix(".gz.part")

            with open(temp, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        # Verify the gzip file before keeping it.
        with gzip.open(temp, "rb") as f:
            f.read(1024)

        temp.replace(archive)
        print(f"Downloaded: {archive}")
        break

    except Exception as exc:
        print(f"Attempt failed: {exc}")

        if attempt == 3:
            raise RuntimeError(
                "Download failed after 3 attempts. "
                "Check internet access and retry."
            ) from exc

        time.sleep(3 * attempt)

# Parse the locally downloaded GEO file.
print("Parsing GEO family file...")
gse = GEOparse.get_GEO(filepath=str(archive))

print("Series:", gse.name)
print("Samples:", len(gse.gsms))
print("Download and parsing completed.")
print("Next: inspect processed sample tables and metadata.")

"""Download IMS bearing dataset from NASA data portal."""
import urllib.request
import zipfile
import shutil
from pathlib import Path

DOWNLOAD_URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip",
    "https://data.nasa.gov/download/brfb-gzcv/application%2Fx-zip-compressed",
]

def download_ims_data(output_dir: str = "data/raw/ims") -> None:
    """Download and extract IMS dataset. ~6GB zip."""
    output_path = Path(output_dir)
    # Check if already extracted
    if (output_path / "1st_test").exists() or (output_path / "1st_test").exists():
        print("IMS data already downloaded.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / "ims_bearing.zip"

    for url in DOWNLOAD_URLS:
        try:
            print(f"Downloading IMS dataset from {url}...")
            print("(This is ~6 GB and may take a while)")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    else:
        raise RuntimeError(
            "Could not download IMS dataset. Try manually:\n"
            "1. Download from https://data.nasa.gov/download/brfb-gzcv/application%2Fx-zip-compressed\n"
            "2. Extract to data/raw/ims/\n"
            "3. Ensure directories 1st_test/, 2nd_test/, 3rd_test/ exist"
        )

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_path)

    # Clean up zip
    zip_path.unlink()
    print(f"IMS data extracted to {output_path}")

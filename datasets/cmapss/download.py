"""Download C-MAPSS dataset from NASA or Kaggle."""
import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_URLS = [
    ("https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip", "nasa_s3"),
    ("https://data.nasa.gov/download/brfb-gzcv/application%2Fx-zip-compressed", "nasa_legacy"),
]

def download_cmapss_data(output_dir: str = "data/raw/cmapss") -> None:
    """Download C-MAPSS dataset."""
    output_path = Path(output_dir)
    if (output_path / "train_FD001.txt").exists():
        print("C-MAPSS data already downloaded.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Try each source
    for url, source in DOWNLOAD_URLS:
        try:
            print(f"Downloading C-MAPSS from {source}...")
            zip_path = output_path / "cmapss.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(output_path)
            zip_path.unlink()

            # Check if files are in a subdirectory
            for subdir in output_path.iterdir():
                if subdir.is_dir() and (subdir / "train_FD001.txt").exists():
                    import shutil
                    for f in subdir.iterdir():
                        shutil.move(str(f), str(output_path / f.name))
                    subdir.rmdir()

            if (output_path / "train_FD001.txt").exists():
                print(f"C-MAPSS data saved to {output_path}")
                return
        except Exception as e:
            print(f"Failed from {source}: {e}")

    raise RuntimeError(
        "Could not download C-MAPSS. Try manually:\n"
        "1. Download from https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository\n"
        "2. Or: kaggle datasets download -d behrad3d/nasa-cmaps\n"
        "3. Extract train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt to data/raw/cmapss/"
    )

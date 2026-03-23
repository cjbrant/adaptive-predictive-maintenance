"""Download XJTU-SY bearing dataset."""
import shutil
import zipfile
from pathlib import Path

# Google Drive folder ID for XJTU-SY dataset
GDRIVE_FOLDER_ID = "1_ycmG46PARiykt82ShfnFfyQsaXv3ua"

# Top-level condition directories present in the dataset
_CONDITION_DIRS = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]


def download_xjtu_sy_data(output_dir: str = "data/raw/xjtu_sy") -> None:
    """Download and extract the XJTU-SY dataset.

    Tries Google Drive via gdown first. Falls back to instructions for
    manual download if automated download fails.
    """
    output_path = Path(output_dir)

    # Check if already extracted
    if any((output_path / d).exists() for d in _CONDITION_DIRS):
        print("XJTU-SY data already downloaded.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Try Google Drive download via gdown
    try:
        import gdown
        print("Downloading XJTU-SY dataset from Google Drive...")
        print("(This is ~1 GB and may take several minutes)")
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=str(output_path),
            quiet=False,
        )

        # gdown may create a nested directory — flatten if needed
        _flatten_nested(output_path)

        if any((output_path / d).exists() for d in _CONDITION_DIRS):
            print(f"XJTU-SY data extracted to {output_path}")
            return
    except Exception as e:
        print(f"Google Drive download failed: {e}")

    # Check if any zip files were downloaded
    for zip_file in output_path.glob("*.zip"):
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(output_path)
        zip_file.unlink()

    _flatten_nested(output_path)

    if any((output_path / d).exists() for d in _CONDITION_DIRS):
        print(f"XJTU-SY data extracted to {output_path}")
        return

    print(
        "WARNING: Could not download XJTU-SY dataset automatically.\n"
        "Please download manually:\n"
        "1. Go to https://drive.google.com/open?id=1_ycmG46PARiykt82ShfnFfyQsaXv3ua\n"
        "2. Download and extract so that 35Hz12kN/, 37.5Hz11kN/, 40Hz10kN/ exist under data/raw/xjtu_sy/\n"
        "The XJTU-SY benchmark will be skipped until the data is available."
    )


def _flatten_nested(output_path: Path) -> None:
    """Move condition directories up if they're nested one level deep."""
    for subdir in list(output_path.iterdir()):
        if not subdir.is_dir():
            continue
        # Check if any condition dirs exist inside this subdir
        for cond_dir in _CONDITION_DIRS:
            nested_cond = subdir / cond_dir
            if nested_cond.exists():
                dest = output_path / cond_dir
                if not dest.exists():
                    shutil.move(str(nested_cond), str(dest))
        # Clean up empty parent
        if subdir.exists() and not any(subdir.iterdir()):
            subdir.rmdir()

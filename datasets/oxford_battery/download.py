"""Download Oxford battery dataset."""
from pathlib import Path
import urllib.request


def download_oxford_battery_data(output_dir: str = "data/raw/oxford_battery") -> None:
    """Download Oxford battery dataset from ORA.

    The dataset is a MATLAB .mat file containing characterization cycle data.
    """
    output_path = Path(output_dir)
    if list(output_path.glob("*.mat")):
        print("Oxford battery data already downloaded.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # ORA URL for the dataset
    url = "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/r5d86p0346"

    try:
        print(f"Downloading Oxford battery dataset...")
        dest = output_path / "oxford_battery.mat"
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Oxford battery data: {e}\n"
            "Try manually from: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac"
        )

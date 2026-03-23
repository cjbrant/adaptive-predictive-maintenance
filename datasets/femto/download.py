"""Download FEMTO/PRONOSTIA dataset from GitHub."""
import subprocess
from pathlib import Path


def download_femto_data(output_dir: str = "data/raw/femto") -> None:
    """Download FEMTO dataset by cloning the GitHub mirror."""
    output_path = Path(output_dir)
    if (output_path / "Learning_set").exists() or (output_path / "learning_set").exists():
        print("FEMTO data already downloaded.")
        return
    output_path.mkdir(parents=True, exist_ok=True)
    repo_url = "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset.git"
    print(f"Cloning FEMTO dataset from {repo_url}...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(output_path)],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone FEMTO dataset: {e.stderr}")
    print(f"FEMTO data saved to {output_path}")

"""Download CWRU Bearing Dataset .mat files.

The Case Western Reserve University Bearing Data Center provides vibration
data from a test rig with seeded bearing faults at various severity levels.
This script downloads the drive-end bearing data at 12 kHz sampling rate.
"""

import os
import urllib.request
from pathlib import Path

# CWRU Bearing Data Center base URL (site migrated to Drupal 10)
BASE_URL = "https://engineering.case.edu/sites/default/files"

# File mappings: descriptive name -> (CWRU file number, description)
# Drive end bearing (SKF 6205-2RS), 12kHz sampling rate
CWRU_FILES = {
    # Normal baseline at different motor loads (HP)
    "normal_0hp": (97, "Normal baseline, 0 HP, 1797 RPM"),
    "normal_1hp": (98, "Normal baseline, 1 HP, 1772 RPM"),
    "normal_2hp": (99, "Normal baseline, 2 HP, 1750 RPM"),
    "normal_3hp": (100, "Normal baseline, 3 HP, 1730 RPM"),
    # Inner race faults - 12kHz drive end
    "ir_007_0hp": (105, "Inner race 0.007 in, 0 HP, 1797 RPM"),
    "ir_007_1hp": (106, "Inner race 0.007 in, 1 HP, 1772 RPM"),
    "ir_007_2hp": (107, "Inner race 0.007 in, 2 HP, 1750 RPM"),
    "ir_007_3hp": (108, "Inner race 0.007 in, 3 HP, 1730 RPM"),
    "ir_014_0hp": (169, "Inner race 0.014 in, 0 HP, 1797 RPM"),
    "ir_014_1hp": (170, "Inner race 0.014 in, 1 HP, 1772 RPM"),
    "ir_014_2hp": (171, "Inner race 0.014 in, 2 HP, 1750 RPM"),
    "ir_014_3hp": (172, "Inner race 0.014 in, 3 HP, 1730 RPM"),
    "ir_021_0hp": (209, "Inner race 0.021 in, 0 HP, 1797 RPM"),
    "ir_021_1hp": (210, "Inner race 0.021 in, 1 HP, 1772 RPM"),
    "ir_021_2hp": (211, "Inner race 0.021 in, 2 HP, 1750 RPM"),
    "ir_021_3hp": (212, "Inner race 0.021 in, 3 HP, 1730 RPM"),
    # Outer race faults - centered, 12kHz drive end
    "or_007_0hp": (130, "Outer race 0.007 in, centered, 0 HP, 1797 RPM"),
    "or_007_1hp": (131, "Outer race 0.007 in, centered, 1 HP, 1772 RPM"),
    "or_007_2hp": (132, "Outer race 0.007 in, centered, 2 HP, 1750 RPM"),
    "or_007_3hp": (133, "Outer race 0.007 in, centered, 3 HP, 1730 RPM"),
    "or_014_0hp": (197, "Outer race 0.014 in, centered, 0 HP, 1797 RPM"),
    "or_014_1hp": (198, "Outer race 0.014 in, centered, 1 HP, 1772 RPM"),
    "or_014_2hp": (199, "Outer race 0.014 in, centered, 2 HP, 1750 RPM"),
    "or_014_3hp": (200, "Outer race 0.014 in, centered, 3 HP, 1730 RPM"),
    "or_021_0hp": (234, "Outer race 0.021 in, centered, 0 HP, 1797 RPM"),
    "or_021_1hp": (235, "Outer race 0.021 in, centered, 1 HP, 1772 RPM"),
    "or_021_2hp": (236, "Outer race 0.021 in, centered, 2 HP, 1750 RPM"),
    "or_021_3hp": (237, "Outer race 0.021 in, centered, 3 HP, 1730 RPM"),
    # Ball faults - 12kHz drive end
    "ball_007_0hp": (118, "Ball fault 0.007 in, 0 HP, 1797 RPM"),
    "ball_007_1hp": (119, "Ball fault 0.007 in, 1 HP, 1772 RPM"),
    "ball_007_2hp": (120, "Ball fault 0.007 in, 2 HP, 1750 RPM"),
    "ball_007_3hp": (121, "Ball fault 0.007 in, 3 HP, 1730 RPM"),
    "ball_014_0hp": (185, "Ball fault 0.014 in, 0 HP, 1797 RPM"),
    "ball_014_1hp": (186, "Ball fault 0.014 in, 1 HP, 1772 RPM"),
    "ball_014_2hp": (187, "Ball fault 0.014 in, 2 HP, 1750 RPM"),
    "ball_014_3hp": (188, "Ball fault 0.014 in, 3 HP, 1730 RPM"),
    "ball_021_0hp": (222, "Ball fault 0.021 in, 0 HP, 1797 RPM"),
    "ball_021_1hp": (223, "Ball fault 0.021 in, 1 HP, 1772 RPM"),
    "ball_021_2hp": (224, "Ball fault 0.021 in, 2 HP, 1750 RPM"),
    "ball_021_3hp": (225, "Ball fault 0.021 in, 3 HP, 1730 RPM"),
}

# RPM for each motor load
LOAD_RPM = {
    "0hp": 1797,
    "1hp": 1772,
    "2hp": 1750,
    "3hp": 1730,
}


def download_file(file_number: int, dest_path: Path) -> bool:
    """Download a single .mat file from the CWRU data center."""
    url = f"{BASE_URL}/{file_number}.mat"
    try:
        print(f"  Downloading {url} -> {dest_path.name}...")
        urllib.request.urlretrieve(url, str(dest_path))
        return True
    except Exception as e:
        print(f"  Failed to download {file_number}: {e}")
        return False


def download_cwru_data(
    output_dir: str | Path = "data/raw",
    subset: str = "all",
    load_hp: int | None = None,
) -> dict[str, Path]:
    """
    Download CWRU bearing dataset files.

    Parameters
    ----------
    output_dir : path to save .mat files
    subset : "all", "normal", "inner_race", "outer_race", "ball",
             or "minimal" (one file per condition for quick testing)
    load_hp : if specified, only download files at this motor load (0, 1, 2, or 3)

    Returns
    -------
    dict mapping descriptive name to local file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter files based on subset
    if subset == "all":
        files = CWRU_FILES
    elif subset == "minimal":
        files = {
            k: v
            for k, v in CWRU_FILES.items()
            if k
            in [
                "normal_0hp",
                "ir_007_0hp",
                "ir_014_0hp",
                "ir_021_0hp",
                "or_007_0hp",
                "or_014_0hp",
                "or_021_0hp",
                "ball_007_0hp",
                "ball_014_0hp",
                "ball_021_0hp",
            ]
        }
    elif subset == "normal":
        files = {k: v for k, v in CWRU_FILES.items() if k.startswith("normal")}
    elif subset == "inner_race":
        files = {k: v for k, v in CWRU_FILES.items() if k.startswith("ir") or k.startswith("normal")}
    elif subset == "outer_race":
        files = {k: v for k, v in CWRU_FILES.items() if k.startswith("or") or k.startswith("normal")}
    elif subset == "ball":
        files = {k: v for k, v in CWRU_FILES.items() if k.startswith("ball") or k.startswith("normal")}
    else:
        raise ValueError(f"Unknown subset: {subset}")

    # Filter by load if specified
    if load_hp is not None:
        load_suffix = f"{load_hp}hp"
        files = {k: v for k, v in files.items() if k.endswith(load_suffix)}

    downloaded = {}
    print(f"Downloading {len(files)} files to {output_dir}/")

    for name, (file_num, description) in files.items():
        dest = output_dir / f"{name}.mat"
        if dest.exists():
            print(f"  Skipping {name} (already exists)")
            downloaded[name] = dest
            continue

        if download_file(file_num, dest):
            downloaded[name] = dest

    print(f"Downloaded {len(downloaded)}/{len(files)} files.")
    return downloaded


def get_mat_key(name: str) -> str:
    """
    Get the MATLAB variable key for drive-end 12kHz data.

    CWRU .mat files use keys like 'X097_DE_time' where 097 is the file number.
    """
    file_num = CWRU_FILES[name][0]
    return f"X{file_num:03d}_DE_time"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CWRU bearing dataset")
    parser.add_argument(
        "--subset",
        default="minimal",
        choices=["all", "minimal", "normal", "inner_race", "outer_race", "ball"],
        help="Which subset of files to download",
    )
    parser.add_argument("--load", type=int, default=None, choices=[0, 1, 2, 3], help="Motor load in HP")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()

    download_cwru_data(args.output, subset=args.subset, load_hp=args.load)

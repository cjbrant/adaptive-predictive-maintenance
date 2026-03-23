"""Oxford battery feature extraction — capacity from characterization cycles."""
import numpy as np
import pandas as pd
from pathlib import Path


def extract_battery_features(mat_data: dict, cell_id: str,
                              nominal_capacity: float = 740.0) -> pd.DataFrame:
    """Extract capacity degradation features from a single cell.

    The mat file contains characterization cycle data. Each characterization
    cycle provides a 1C discharge capacity measurement.

    Returns DataFrame with columns: cycle, capacity_mah, soh
    """
    # Try different key patterns the mat file might use
    possible_keys = [
        cell_id, f"cell_{cell_id}", f"Cell_{cell_id}",
        f"data_{cell_id}", cell_id.upper(), cell_id.lower()
    ]

    cell_data = None
    for key in possible_keys:
        if key in mat_data:
            cell_data = mat_data[key]
            break

    if cell_data is None:
        # Try to find any key matching the cell pattern
        for key in mat_data.keys():
            if cell_id.lower() in key.lower():
                cell_data = mat_data[key]
                break

    if cell_data is None:
        raise KeyError(f"Could not find data for cell {cell_id} in mat file. "
                       f"Available keys: {list(mat_data.keys())}")

    # Extract capacity values - structure depends on mat file format
    if hasattr(cell_data, 'dtype') and cell_data.dtype.names:
        # Structured array
        if 'capacity' in cell_data.dtype.names:
            capacities = cell_data['capacity'].flatten()
        elif 'Capacity' in cell_data.dtype.names:
            capacities = cell_data['Capacity'].flatten()
        else:
            # Try first numeric field
            for name in cell_data.dtype.names:
                try:
                    vals = cell_data[name].flatten()
                    if len(vals) > 10 and np.issubdtype(vals.dtype, np.number):
                        capacities = vals
                        break
                except Exception:
                    continue
            else:
                raise ValueError(f"Could not find capacity data in cell {cell_id}")
    elif isinstance(cell_data, np.ndarray):
        if cell_data.ndim == 2:
            # Assume capacity is in one of the columns
            capacities = cell_data[:, 0] if cell_data.shape[1] > 0 else cell_data.flatten()
        else:
            capacities = cell_data.flatten()
    else:
        raise ValueError(f"Unexpected data format for cell {cell_id}")

    cycles = np.arange(1, len(capacities) + 1)
    soh = capacities / nominal_capacity

    df = pd.DataFrame({
        "cycle": cycles,
        "capacity_mah": capacities,
        "soh": soh,
    })
    df = df.set_index("cycle")
    return df


def extract_all_cells(mat_path: str, nominal_capacity: float = 740.0,
                       n_cells: int = 8) -> dict[str, pd.DataFrame]:
    """Extract capacity data for all cells from the mat file."""
    from scipy.io import loadmat

    mat_data = loadmat(mat_path, simplify_cells=True)

    # Remove MATLAB metadata keys
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]

    results = {}
    for key in data_keys:
        try:
            df = extract_battery_features(mat_data, key, nominal_capacity)
            if len(df) > 10:  # Only keep cells with reasonable data
                results[key] = df
                print(f"  Extracted {key}: {len(df)} cycles, "
                      f"SOH range [{df['soh'].min():.3f}, {df['soh'].max():.3f}]")
        except Exception as e:
            print(f"  Skipping {key}: {e}")

    return results

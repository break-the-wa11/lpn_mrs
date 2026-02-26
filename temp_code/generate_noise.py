import os
from pathlib import Path
import numpy as np
import pandas as pd

# SETTINGS - change these to your needs
source_root = Path('./data/test')      # parent of folders like low_lipid
target_root = Path('./data/noise')     # parent where noise/<sigma>/<subfolder> will be created
subfolder = 'low_lipid'                # the data_type folder to process
sigma_values = [0.05, 0.1, 0.15, 0.2]                    # list of sigma(s) to generate; can be multiple
file_glob = 'Spec_*.tsv'                # pattern for data files
seed = 0                                # rng seed for reproducibility

# Other options
delimiter = '\t'
ppm_col_name = 'ppm'                    # column name in input to copy
float_format = '%.14f'                  # plain decimal formatting with 14 digits after decimal

# Initialize RNG
rng = np.random.default_rng(seed)

# Compose source and check
source_dir = source_root / subfolder
if not source_dir.exists():
    raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

# Collect files
tsv_files = sorted(source_dir.glob(file_glob))
if not tsv_files:
    raise FileNotFoundError(f"No files found matching {file_glob} in {source_dir}")

print(f"Found {len(tsv_files)} files in {source_dir}")

# Function to generate and save noise file for a single sigma and source file
def generate_noise_file(src_path: Path, sigma: float, out_dir: Path):
    # Read input TSV (assumes header row present)
    try:
        df = pd.read_csv(src_path, sep=delimiter, header=0)
    except Exception as e:
        raise RuntimeError(f"Failed to read {src_path}: {e}")

    if ppm_col_name not in df.columns:
        raise ValueError(f"Column '{ppm_col_name}' not found in {src_path}. Columns: {df.columns.tolist()}")

    ppm_values = df[ppm_col_name].to_numpy()

    # Generate noise arrays for FD_Re and FD_Im
    n = ppm_values.shape[0]
    fd_re = rng.normal(loc=0.0, scale=sigma, size=n)
    fd_im = rng.normal(loc=0.0, scale=sigma, size=n)

    # Build output dataframe with required header and order
    out_df = pd.DataFrame({
        'ppm': ppm_values,
        'FD_Re': fd_re,
        'FD_Im': fd_im
    })

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / src_path.name
    # Save as TSV with header and no extra index column, using decimal float format
    out_df.to_csv(out_path, sep=delimiter, index=False, float_format=float_format)
    return out_path

# Iterate over sigma values and source files
for sigma in sigma_values:
    for src in tsv_files:
        out_folder = target_root / subfolder / f"{sigma}" 
        out_file = generate_noise_file(src, sigma, out_folder)
        print(f"Wrote: {out_file}")

print("Done.")
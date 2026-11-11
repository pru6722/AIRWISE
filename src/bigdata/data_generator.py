"""
Data Generator for Scalability Benchmarks (Phase 4).
Expands original 18,584 observations to 1x, 5x, 10x, 25x, and 50x scales.
Ensures explicit synthetic provenance, unique record IDs, valid physical bounds,
and time-shifted block resampling.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


# Physiological and sensor validity bounds
BOUNDS = {
    "bam_reference": (0.0, 1500.0),
    "pa_cf1": (0.0, 2000.0),
    "pa_atm": (0.0, 2000.0),
    "pa_alt": (0.0, 2000.0),
    "relative_humidity": (0.0, 100.0),
    "temperature": (-10.0, 65.0),
    "dew_point_raw": (-50.0, 150.0),
}


def generate_benchmark_datasets(
    input_path: str = "data/processed/cleaned_data.parquet",
    output_dir: str = "data/benchmark_scales",
    scales: list = [1, 5, 10, 25, 50],
    seed: int = 42,
    save_csv: bool = True,
    save_parquet: bool = True,
) -> dict:
    """
    Generates deterministic benchmark datasets at specified expansion scales.
    
    Research Integrity Guard:
    - Expanded records contain `is_synthetic_benchmark = True`
    - Preserves site labels and sensor distributions
    - Never replaces or alters original cleaned dataset
    """
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Original dataset not found at {input_path}")

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_parquet(in_file)
    n_orig = len(df_orig)
    print(f"Loaded original dataset with {n_orig:,} observations.")

    summary_catalog = {}

    for scale in scales:
        target_rows = n_orig * scale
        print(f"\n--- Generating {scale}× Benchmark Dataset (~{target_rows:,} rows) ---")

        dfs = []
        # Block 0 is the original reference
        df_block0 = df_orig.copy()
        df_block0["record_id"] = [f"REC_{scale}X_0_{i:08d}" for i in range(n_orig)]
        df_block0["replication_block"] = 0
        df_block0["expansion_scale"] = scale
        df_block0["is_synthetic_benchmark"] = (scale > 1)  # 1x is authentic real data
        dfs.append(df_block0)

        # Generate subsequent blocks with deterministic time shift and realistic perturbation
        for b in range(1, scale):
            df_b = df_orig.copy()
            df_b["record_id"] = [f"REC_{scale}X_{b}_{i:08d}" for i in range(n_orig)]
            df_b["replication_block"] = b
            df_b["expansion_scale"] = scale
            df_b["is_synthetic_benchmark"] = True

            # Deterministic time shift: shift by b years (365 * 24 hours)
            time_shift_hours = b * 365 * 24
            df_b["timestamp"] = df_b["timestamp"] + pd.Timedelta(hours=time_shift_hours)

            # Add minor pseudo-random perturbation (mean 0, 0.5 std)
            block_seed = seed + b * 10007
            block_rng = np.random.default_rng(block_seed)
            for col, (min_v, max_v) in BOUNDS.items():
                if col in df_b.columns:
                    noise = block_rng.normal(0.0, 0.5, size=n_orig).astype(np.float32)
                    df_b[col] = np.clip(df_b[col] + noise, min_v, max_v)

            dfs.append(df_b)

        df_scale = pd.concat(dfs, ignore_index=True)
        # Ensure timestamp is datetime64[us] for full Parquet / Spark interoperability
        df_scale["timestamp"] = pd.to_datetime(df_scale["timestamp"]).astype("datetime64[us]")
        df_scale["site"] = df_scale["site"].astype(str)

        # File paths
        pq_path = out_base / f"scale_{scale}x.parquet"
        csv_path = out_base / f"scale_{scale}x.csv"

        if save_parquet:
            df_scale.to_parquet(pq_path, index=False)
            pq_size_mb = pq_path.stat().st_size / (1024 * 1024)
        else:
            pq_size_mb = 0.0

        if save_csv:
            df_scale.to_csv(csv_path, index=False)
            csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        else:
            csv_size_mb = 0.0

        summary_catalog[scale] = {
            "scale": scale,
            "total_rows": len(df_scale),
            "parquet_path": str(pq_path),
            "parquet_size_mb": round(pq_size_mb, 2),
            "csv_path": str(csv_path),
            "csv_size_mb": round(csv_size_mb, 2),
            "real_rows": n_orig if scale == 1 else 0,
            "synthetic_rows": 0 if scale == 1 else len(df_scale),
            "sites": df_scale["site"].value_counts().to_dict(),
        }
        print(f"Scale {scale}× created: {len(df_scale):,} rows | Parquet: {pq_size_mb:.2f} MB | CSV: {csv_size_mb:.2f} MB")

    return summary_catalog


if __name__ == "__main__":
    generate_benchmark_datasets()

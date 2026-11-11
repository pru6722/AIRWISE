"""
Module: src.data.ingestion
Loads and standardizes multi-site air quality sensor datasets from the Campmier manuscript Excel workbook.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import yaml


DEFAULT_COLUMN_MAPPING = {
    "Index": "timestamp",
    "BAM (ug/m3)": "bam_reference",
    "PA_CF1 (ug/m3)": "pa_cf1",
    "PA_ATM (ug/m3)": "pa_atm",
    "PA_ALT (ug/m3)": "pa_alt",
    "PA_RH (%)": "relative_humidity",
    "PA_Temperature (°C)": "temperature",
    "PA_DewPoint (°C)": "dew_point_raw",
}


def load_raw_workbook(
    excel_path: Union[str, Path],
    sheet_names: Optional[List[str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Reads all sheets from the Campmier manuscript Excel workbook, standardizes column
    names, assigns the site identifier, and returns both unified and per-site DataFrames.

    Args:
        excel_path: Path to the raw Excel workbook (.xlsx).
        sheet_names: List of sheet names to load (defaults to ['Delhi', 'Hamirpur', 'Bangalore']).
        column_mapping: Mapping from Excel column names to research standard names.

    Returns:
        Tuple of (combined_df, dict_of_site_dfs).
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Workbook not found at {excel_path.resolve()}")

    if column_mapping is None:
        column_mapping = DEFAULT_COLUMN_MAPPING

    xl = pd.ExcelFile(excel_path)
    available_sheets = xl.sheet_names

    if sheet_names is None:
        sheet_names = [s for s in ["Delhi", "Hamirpur", "Bangalore"] if s in available_sheets]
        if not sheet_names:
            sheet_names = available_sheets

    site_dfs: Dict[str, pd.DataFrame] = {}
    combined_rows: List[pd.DataFrame] = []

    for site in sheet_names:
        raw_sheet = xl.parse(site)
        df = raw_sheet.rename(columns=column_mapping).copy()

        # Enforce column schema
        required_cols = list(column_mapping.values())
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Sheet '{site}' is missing expected standardized columns: {missing_cols}")

        # Standardize timestamp and site
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["site"] = site

        # Ensure correct column ordering
        standard_cols = [
            "timestamp",
            "site",
            "bam_reference",
            "pa_cf1",
            "pa_atm",
            "pa_alt",
            "relative_humidity",
            "temperature",
            "dew_point_raw",
        ]
        df = df[standard_cols].sort_values("timestamp").reset_index(drop=True)

        site_dfs[site] = df
        combined_rows.append(df)

    combined_df = pd.concat(combined_rows, ignore_index=True)
    return combined_df, site_dfs


def load_from_config(config_path: Union[str, Path] = "configs/data_config.yaml") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Loads raw dataset using parameters from configuration YAML file."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg.get("raw_data_path", "data/raw/Campmier_Manuscript_Data.xlsx")
    sites = cfg.get("sites", ["Delhi", "Hamirpur", "Bangalore"])
    col_map = cfg.get("column_mapping", DEFAULT_COLUMN_MAPPING)

    return load_raw_workbook(raw_path, sheet_names=sites, column_mapping=col_map)

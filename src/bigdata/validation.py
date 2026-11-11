"""
Validation and Correctness Module for Checkpoint 6.
Compares Spark and Pandas outputs across all scales and metrics.
Enforces numerical tolerances and verifies complete consistency.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame


class BigDataValidator:
    """Verifies numerical consistency between Spark and Pandas pipelines."""

    def __init__(self, float_tolerance: float = 1e-3):
        self.tol = float_tolerance

    def validate_descriptive_stats(
        self,
        spark_stats_df: SparkDataFrame,
        pandas_stats_df: pd.DataFrame,
    ) -> Tuple[bool, List[str]]:
        """Compares descriptive stats across sites."""
        errors = []
        spark_pdf = spark_stats_df.toPandas()

        if len(spark_pdf) != len(pandas_stats_df):
            errors.append(f"Site count mismatch: Spark={len(spark_pdf)}, Pandas={len(pandas_stats_df)}")
            return False, errors

        # Sort both by site
        spark_pdf = spark_pdf.sort_values("site").reset_index(drop=True)
        pandas_pdf = pandas_stats_df.sort_values("site").reset_index(drop=True)

        for i, row_s in spark_pdf.iterrows():
            row_p = pandas_pdf.iloc[i]
            site_s = row_s["site"]
            site_p = row_p["site"]

            if site_s != site_p:
                errors.append(f"Site name mismatch at index {i}: {site_s} vs {site_p}")
                continue

            # Check obs count
            count_s = int(row_s["obs_count"])
            # In pandas_pdf, could be 'bam_reference_count' or 'obs_count'
            count_p = int(row_p.get("obs_count", row_p.get("bam_reference_count", 0)))
            if count_s != count_p:
                errors.append(f"Count mismatch for site {site_s}: Spark={count_s}, Pandas={count_p}")

            # Check mean values
            for ch in ["bam_reference", "pa_atm", "pa_cf1", "pa_alt", "relative_humidity", "temperature"]:
                s_mean = float(row_s[f"{ch}_mean"])
                p_mean = float(row_p[f"{ch}_mean"])
                diff = abs(s_mean - p_mean)
                if diff > self.tol:
                    errors.append(
                        f"Mean mismatch for {site_s}.{ch}: Spark={s_mean:.4f}, Pandas={p_mean:.4f}, diff={diff:.6f}"
                    )

        passed = len(errors) == 0
        return passed, errors

    def validate_daily_aggregations(
        self,
        spark_daily_df: SparkDataFrame,
        pandas_daily_df: pd.DataFrame,
    ) -> Tuple[bool, List[str]]:
        """Compares daily 24-hour aggregation outputs."""
        errors = []
        spark_daily = spark_daily_df.toPandas()

        if len(spark_daily) != len(pandas_daily_df):
            errors.append(
                f"Daily aggregate row count mismatch: Spark={len(spark_daily)}, Pandas={len(pandas_daily_df)}"
            )
            return False, errors

        # Convert date to string for consistent comparison
        spark_daily["date_str"] = spark_daily["date"].astype(str)
        pandas_daily_df = pandas_daily_df.copy()
        pandas_daily_df["date_str"] = pandas_daily_df["date"].astype(str)

        spark_daily = spark_daily.sort_values(["site", "date_str"]).reset_index(drop=True)
        pandas_daily_df = pandas_daily_df.sort_values(["site", "date_str"]).reset_index(drop=True)

        for i, row_s in spark_daily.iterrows():
            row_p = pandas_daily_df.iloc[i]
            site = row_s["site"]
            dt = row_s["date_str"]

            if row_s["site"] != row_p["site"] or dt != row_p["date_str"]:
                errors.append(f"Key mismatch at {i}: Spark=({row_s['site']}, {dt}) vs Pandas=({row_p['site']}, {row_p['date_str']})")
                break

            # Compare valid hours
            if int(row_s["valid_hours"]) != int(row_p["valid_hours"]):
                errors.append(f"Valid hours mismatch at {site} {dt}: Spark={row_s['valid_hours']} vs Pandas={row_p['valid_hours']}")

            # Compare BAM and PA means
            for col in ["bam_24h_mean", "pa_atm_24h_mean"]:
                diff = abs(float(row_s[col]) - float(row_p[col]))
                if diff > self.tol:
                    errors.append(f"{col} mismatch at {site} {dt}: Spark={row_s[col]} vs Pandas={row_p[col]}, diff={diff:.6f}")

            # Compare categories
            if str(row_s["cpcb_ref_category"]) != str(row_p["cpcb_ref_category"]):
                errors.append(f"CPCB category mismatch at {site} {dt}: Spark={row_s['cpcb_ref_category']} vs Pandas={row_p['cpcb_ref_category']}")

            if len(errors) > 10:
                errors.append("Truncating further row error checks...")
                break

        passed = len(errors) == 0
        return passed, errors

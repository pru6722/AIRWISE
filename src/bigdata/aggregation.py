"""
24-Hour Daily Aggregation Module for Checkpoint 6.
Implements CPCB category assignment and regulatory 24-hour aggregation (75% completeness rule,
>= 18 valid hours per day) strictly partitioned by site without bridging gaps.
"""

from typing import Tuple
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


# Continuous CPCB Category Logic in Spark SQL expression
CPCB_SPARK_CASE_BAM = """
CASE 
    WHEN bam_24h_mean <= 30.0 THEN 'Good'
    WHEN bam_24h_mean <= 60.0 THEN 'Satisfactory'
    WHEN bam_24h_mean <= 90.0 THEN 'Moderate'
    WHEN bam_24h_mean <= 120.0 THEN 'Poor'
    WHEN bam_24h_mean <= 250.0 THEN 'Very Poor'
    ELSE 'Severe'
END
"""

CPCB_SPARK_CASE_PA = """
CASE 
    WHEN pa_atm_24h_mean <= 30.0 THEN 'Good'
    WHEN pa_atm_24h_mean <= 60.0 THEN 'Satisfactory'
    WHEN pa_atm_24h_mean <= 90.0 THEN 'Moderate'
    WHEN pa_atm_24h_mean <= 120.0 THEN 'Poor'
    WHEN pa_atm_24h_mean <= 250.0 THEN 'Very Poor'
    ELSE 'Severe'
END
"""


def assign_cpcb_category_scalar(conc: float) -> str:
    """Assigns official continuous CPCB category to a PM2.5 concentration value."""
    if pd.isna(conc):
        return "Unknown"
    if conc <= 30.0:
        return "Good"
    elif conc <= 60.0:
        return "Satisfactory"
    elif conc <= 90.0:
        return "Moderate"
    elif conc <= 120.0:
        return "Poor"
    elif conc <= 250.0:
        return "Very Poor"
    else:
        return "Severe"


# ---------------------------------------------------------
# SPARK 24-HOUR AGGREGATION
# ---------------------------------------------------------

def aggregate_24h_spark(
    df: SparkDataFrame,
    min_hours: int = 18,
) -> SparkDataFrame:
    """
    Computes daily 24-hour averages per site, filtering by completeness (>= min_hours).
    Assigns CPCB categories based on daily mean PM2.5.
    """
    df_daily = (
        df.withColumn("date", F.to_date("timestamp"))
        .groupBy("site", "date")
        .agg(
            F.count("*").alias("valid_hours"),
            F.round(F.mean("bam_reference"), 4).alias("bam_24h_mean"),
            F.round(F.mean("pa_atm"), 4).alias("pa_atm_24h_mean"),
            F.round(F.mean("pa_cf1"), 4).alias("pa_cf1_24h_mean"),
            F.round(F.mean("pa_alt"), 4).alias("pa_alt_24h_mean"),
            F.round(F.mean("relative_humidity"), 4).alias("rh_24h_mean"),
            F.round(F.mean("temperature"), 4).alias("temp_24h_mean"),
        )
        .filter(F.col("valid_hours") >= min_hours)
        .withColumn("cpcb_ref_category", F.expr(CPCB_SPARK_CASE_BAM))
        .withColumn("cpcb_sensor_category", F.expr(CPCB_SPARK_CASE_PA))
        .orderBy("site", "date")
    )
    return df_daily


# ---------------------------------------------------------
# PANDAS 24-HOUR AGGREGATION
# ---------------------------------------------------------

def aggregate_24h_pandas(
    df: pd.DataFrame,
    min_hours: int = 18,
) -> pd.DataFrame:
    """
    Computes daily 24-hour averages per site in Pandas with identical completeness logic.
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    agg_dict = {
        "timestamp": "count",
        "bam_reference": "mean",
        "pa_atm": "mean",
        "pa_cf1": "mean",
        "pa_alt": "mean",
        "relative_humidity": "mean",
        "temperature": "mean",
    }

    daily = df.groupby(["site", "date"], observed=False).agg(agg_dict).reset_index()
    daily = daily.rename(columns={
        "timestamp": "valid_hours",
        "bam_reference": "bam_24h_mean",
        "pa_atm": "pa_atm_24h_mean",
        "pa_cf1": "pa_cf1_24h_mean",
        "pa_alt": "pa_alt_24h_mean",
        "relative_humidity": "rh_24h_mean",
        "temperature": "temp_24h_mean",
    })

    # Filter completeness
    daily = daily[daily["valid_hours"] >= min_hours].copy()

    # Round to match Spark
    for col in ["bam_24h_mean", "pa_atm_24h_mean", "pa_cf1_24h_mean", "pa_alt_24h_mean", "rh_24h_mean", "temp_24h_mean"]:
        daily[col] = daily[col].round(4)

    # Assign CPCB categories
    daily["cpcb_ref_category"] = daily["bam_24h_mean"].apply(assign_cpcb_category_scalar)
    daily["cpcb_sensor_category"] = daily["pa_atm_24h_mean"].apply(assign_cpcb_category_scalar)

    daily = daily.sort_values(["site", "date"]).reset_index(drop=True)
    return daily

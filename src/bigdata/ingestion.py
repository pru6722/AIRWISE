"""
Data Ingestion Module for Checkpoint 6.
Handles schema specification, timestamp parsing, and ingestion across CSV, Parquet, and ORC
for both Apache Spark and Pandas.
"""

from pathlib import Path
import pandas as pd


# Explicit PySpark Schema definition
def get_spark_schema():
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        TimestampType,
        FloatType,
        IntegerType,
        BooleanType,
    )
    return StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("site", StringType(), True),
        StructField("bam_reference", FloatType(), True),
        StructField("pa_cf1", FloatType(), True),
        StructField("pa_atm", FloatType(), True),
        StructField("pa_alt", FloatType(), True),
        StructField("relative_humidity", FloatType(), True),
        StructField("temperature", FloatType(), True),
        StructField("dew_point_raw", FloatType(), True),
        StructField("is_dew_point_suspicious", IntegerType(), True),
        StructField("time_diff_hours", FloatType(), True),
        StructField("is_contiguous_step", IntegerType(), True),
        StructField("segment_id", IntegerType(), True),
        StructField("record_id", StringType(), True),
        StructField("replication_block", IntegerType(), True),
        StructField("expansion_scale", IntegerType(), True),
        StructField("is_synthetic_benchmark", BooleanType(), True),
    ])


def ingest_with_pandas(file_path: str, format: str = "parquet") -> pd.DataFrame:
    """Reads dataset using Pandas with standard type parsing."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if format == "parquet":
        df = pd.read_parquet(p)
    elif format == "csv":
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError(f"Unsupported format for Pandas ingestion: {format}")

    # Ensure timestamp is datetime64
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df


def ingest_with_spark(spark, file_path: str, format: str = "parquet", schema=None):
    """Reads dataset using Apache Spark DataFrame reader."""
    p = str(Path(file_path).resolve())
    reader = spark.read

    if schema is not None:
        reader = reader.schema(schema)

    if format == "parquet":
        df = reader.parquet(p)
    elif format == "csv":
        df = reader.option("header", "true").option("timestampFormat", "yyyy-MM-dd HH:mm:ss").csv(p)
    elif format == "orc":
        df = reader.orc(p)
    else:
        raise ValueError(f"Unsupported format for Spark ingestion: {format}")

    return df

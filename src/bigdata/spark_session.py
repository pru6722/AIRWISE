"""
SparkSession factory and lifecycle manager for Checkpoint 6.
Configures Java environment, local multi-core master, warehouse dir, and adaptive query execution.
"""

import os
import sys
from pathlib import Path


def configure_environment():
    """Ensure environment is cleanly configured for PySpark 4.3.0."""
    # Unset conflicting external Hadoop configs that trigger permission errors
    for k in ["HADOOP_HOME", "HADOOP_INSTALL", "HADOOP_CONF_DIR", "YARN_CONF_DIR"]:
        if k in os.environ:
            del os.environ[k]

    # Ensure system Java 24 (or 17+) is used for PySpark 4.3.0
    java_home = os.environ.get("JAVA_HOME")
    if not java_home or not os.path.exists(java_home):
        default_jdk24 = "/Library/Java/JavaVirtualMachines/jdk-24.jdk/Contents/Home"
        if os.path.exists(default_jdk24):
            os.environ["JAVA_HOME"] = default_jdk24


def get_spark_session(
    app_name: str = "PM25_Research_Scalability",
    master: str = "local[*]",
    driver_memory: str = "4g",
    shuffle_partitions: int = 8,
    enable_aqe: bool = True,
    warehouse_dir: str = "outputs/spark-warehouse",
    enable_hive: bool = True,
):
    """Creates or returns an active SparkSession with custom settings."""
    configure_environment()
    
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "pyspark is not installed. Please install PySpark to run Spark operations."
        )

    # Ensure warehouse and derby dirs exist
    wh_path = Path(warehouse_dir).resolve()
    wh_path.mkdir(parents=True, exist_ok=True)
    derby_dir = Path("/tmp/spark_derby").resolve()
    derby_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true" if enable_aqe else "false")
        .config("spark.sql.warehouse.dir", str(wh_path))
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={derby_dir}")
        .config("spark.ui.enabled", "false")  # Disable web UI for batch benchmarking
        .config("spark.sql.session.timeZone", "UTC")
    )

    if enable_hive:
        try:
            builder = builder.enableHiveSupport()
        except Exception as e:
            print(f"Warning: Hive support could not be enabled: {e}. Falling back to standard catalog.")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark_session(spark):
    """Safely stop Spark session and clear JVM resources."""
    if spark is not None:
        try:
            spark.stop()
        except Exception as e:
            print(f"Error stopping SparkSession: {e}")

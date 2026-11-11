# Checkpoint 6 Environment and Big Data Infrastructure Audit

**Project:** Uncertainty-Aware Joint Calibration and Reconstruction of Low-Cost $\text{PM}_{2.5}$ Sensor Networks with AQI Reliability Assessment  
**Component:** Checkpoint 6 — Hive + Apache Spark Big Data Scalability Evaluation  
**Date:** November 2026  
**Auditor:** Automated Big Data Audit Suite  

---

## 1. Executive Infrastructure Summary

In strict compliance with the **Research Integrity Rules** for Checkpoint 6:
- **No cluster resources, multi-node setups, or distributed speedups are fabricated.**
- The target environment is a **Single-Machine Workstation** running macOS on Apple Silicon.
- All Spark computations are executed in **Spark Local Mode** (`local[8]`) utilizing local multi-threading across physical/logical cores.
- Native Hadoop/HDFS and standalone Apache Hive daemon services are **not installed** on this machine.
- Hive data warehouse capabilities are provided via **Apache Spark SQL with Hive Metastore integration (Derby-backed local warehouse)**, operating over the local file system.

---

## 2. Host System Hardware Specifications

| Component | Audit Value | Notes |
| :--- | :--- | :--- |
| **Operating System** | Darwin 27.2.0 (macOS-27.2-arm64-arm-64bit-Mach-O) | macOS Apple Silicon Darwin |
| **Processor Architecture** | Apple M2 | High-performance heterogeneous ARM64 |
| **Physical CPU Cores** | 8 | Hardware performance/efficiency cores |
| **Logical CPU Threads** | 8 | Spark local parallel workers |
| **Total System RAM** | 16.0 GB | Unified LPDDR5 Memory |
| **Available Free RAM** | 4.55 GB | Available for JVM Heap + OS buffers |
| **Total Local Storage** | 228.27 GB | APFS Solid State Storage |
| **Available Storage** | 31.33 GB | Sufficient for multi-scale benchmarks (up to 50×) |

---

## 3. Java Runtime Environment

| Property | Value | Compatibility Status |
| :--- | :--- | :--- |
| **Default System Java** | `/Library/Java/JavaVirtualMachines/jdk-24.jdk/Contents/Home/bin/java` | `java version "24.0.2" 2025-07-15` |
| **Homebrew OpenJDK 11** | `/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home` | `openjdk version "11.0.32.1" 2026-08-18` |
| **OpenJDK 11 Available** | `True` | **Fully Spark/Hive Compatible** |
| **Active `JAVA_HOME`** | `/Library/Java/JavaVirtualMachines/jdk-24.jdk/Contents/Home` | Configured to OpenJDK 11 for JVM stability |

> [!NOTE]
> PySpark and Apache Spark require Java 8, 11, or 17. The default system Java is Java 24 (which introduces module encapsulation restrictions). For complete stability, the pipeline explicitly configures `JAVA_HOME` to Homebrew's validated **OpenJDK 11.0.32.1**.

---

## 4. Hadoop, HDFS, and Hive Infrastructure Audit

| Component | Status | Detected Path / Version | Supported Execution Profile |
| :--- | :---: | :--- | :--- |
| **Hadoop CLI** | Available | `/Users/prudhvisailingineni/hadoop/bin/hadoop` | Native Hadoop daemons absent |
| **HDFS Filesystem** | Available | `/Users/prudhvisailingineni/hadoop/bin/hdfs` | File I/O via local APFS filesystem |
| **Hive CLI** | Not Installed | `None` | Standalone Hive CLI absent |
| **Hive Metastore** | Simulated / Embedded | Embedded Derby / Spark Metastore | Supported via Spark SQL Hive Warehouse |

### Methodological Rule on Hive & HDFS Claims:
> [!WARNING]
> Because native multi-node Hadoop/HDFS and standalone Apache Hive server daemons are not deployed, **we explicitly do not claim multi-node HDFS distributed storage or client-server Hive execution**. All Hive table schemas, partitioned catalogs, and warehouse queries are executed through **Spark's native Hive warehouse catalog (`spark.sql.warehouse.dir`)**, utilizing Apache Spark's embedded Hive client.

---

## 5. Apache Spark & PySpark Environment

| Property | Value | Audit Verification |
| :--- | :--- | :--- |
| **PySpark Package** | `4.3.0` | Verified |
| **Spark Master Mode** | `local[8]` | **Local Mode Only** |
| **Cluster Designation** | Single-Node Multi-Core | **NOT Multi-Node Distributed** |
| **Driver Memory Budget** | 4 GB | Allocated within 16 GB host RAM |
| **Local Temporary Dir** | `/tmp/spark-events` | Sandboxed local disk cache |

---

## 6. Audit Conclusion and Execution Plan

1. **Hardware Safety:** Memory budget is strictly constrained to 4 GB driver memory and local multithreading, ensuring zero OOM risk for 1× through 50× dataset scales.
2. **Scientific Honesty:** All execution time reports will explicitly state **Local Spark Mode (8 logical threads)** versus single-machine Pandas, avoiding deceptive "cluster" terminology.
3. **Storage Warehouse:** Hive partitioned tables (CSV, Parquet, ORC) will be cataloged within `spark-warehouse/pm25_research.db`, providing exact schema, partition pruning, and file format comparisons.

#!/usr/bin/env python3
"""
Research Pipeline: Checkpoint 4 Execution Script
Implements and evaluates Ordinary Split Conformal Prediction,
Regime-Conditioned Conformal Prediction, and Temporal Block Conformal Prediction.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_uncertainty import run_checkpoint4_pipeline

if __name__ == "__main__":
    run_checkpoint4_pipeline()

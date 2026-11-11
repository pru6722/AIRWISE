"""
Script: src.training.train_joint_tcn_v2
Executes the approved Phase 3C.1B training protocol for Joint TCN v2:
1. Enforces mandatory pre-training causality verification gate (Test A & Test B).
2. Fits only on train_df (11,149 rows, 1,906 continuous 64h windows), early stops on val_model_df (1,857 rows, 489 windows).
3. Keeps conformal calibration and test partitions completely untouched.
4. Uses ChannelLayerNorm1d, K=3, dilations=[1, 2, 4, 8], seq_len=64 (RF=61h).
5. Saves new checkpoint exclusively to outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt.
6. Verifies preservation of historical checkpoints via SHA-256 hashes.
"""

from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch

from src.models.joint_tcn import JointTCN, verify_full_model_causality
from src.training.train_joint_tcn import train_joint_tcn_model, INPUT_FEATURE_COLS
from src.uncertainty.split_manager import create_conformal_splits
from src.evaluation.evaluate_joint_tcn import predict_joint_tcn_full_sequence
from src.evaluation.metrics import compute_regression_metrics


HISTORICAL_CHECKPOINTS = {
    "frozen_joint_tcn": Path("outputs/models/conformal/frozen_joint_tcn_checkpoint.pt"),
    "best_joint_tcn": Path("outputs/models/joint_tcn/best_joint_tcn_checkpoint.pt"),
}

NEW_CHECKPOINT_PATH = Path("outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt")
METADATA_PATH = Path("outputs/models/joint_tcn/joint_tcn_v2_metadata.json")


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("AIRWISE — Phase 3C.1B: Joint TCN v2 Causality Gate & Training Execution")
    print("=" * 80)

    # 1. Artifact Governance: Pre-Execution Hashes
    pre_hashes = {}
    for name, path in HISTORICAL_CHECKPOINTS.items():
        if path.exists():
            pre_hashes[name] = compute_sha256(path)
            print(f"[PRE-HASH] {name} ({path}): {pre_hashes[name]}")
        else:
            print(f"[PRE-HASH] WARNING: {name} not found at {path}")

    # 2. Load dataset and create four-way conformal splits
    data_path = Path("data/processed/cleaned_data.parquet")
    print(f"\n[DATA] Loading cleaned dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"[DATA] Total dataset size: {len(df):,} rows")

    train_df, val_model_df, calib_conformal_df, test_df, split_info = create_conformal_splits(df)
    print(f"[SPLITS] train_df:            {len(train_df):,} rows (used for model fitting)")
    print(f"[SPLITS] val_model_df:        {len(val_model_df):,} rows (used for early stopping)")
    print(f"[SPLITS] calib_conformal_df:  {len(calib_conformal_df):,} rows (FROZEN — Conformal Calibration)")
    print(f"[SPLITS] test_df:             {len(test_df):,} rows (FROZEN — Final Evaluation)")

    # 3. Mandatory Pre-Training Causality Verification Gate
    print("\n" + "=" * 80)
    print("MANDATORY PRE-TRAINING CAUSALITY VERIFICATION GATE")
    print("=" * 80)
    feature_cols = [c for c in INPUT_FEATURE_COLS if c in train_df.columns]
    input_dim = len(feature_cols) + 1  # 10 features + 1 mask channel = 11

    temp_model = JointTCN(
        input_dim=input_dim,
        hidden_channels=32,
        dilations=[1, 2, 4, 8],
        kernel_size=3,
        dropout=0.15,
        norm_type="layer",
    )

    t0_start = time.time()
    causality_report = verify_full_model_causality(
        model=temp_model,
        seq_len=64,
        cut_positions=[16, 32, 48, 60],
        perturbation_magnitudes=[0.1, 10.0, 100.0, 500.0, -50.0],
        tolerance=1e-6,
        batch_size=4,
        random_seed=42,
    )
    causality_duration = time.time() - t0_start

    print(f"[GATE] Completed {causality_report['total_checks']} checks in {causality_duration:.2f}s")
    print(f"[GATE] Test A (Value Perturbation across magnitudes): {len(causality_report['test_a_results'])} checks")
    print(f"[GATE] Test B (Mask Perturbation with fixed features): {len(causality_report['test_b_results'])} checks")
    print(f"[GATE] Max past deviation: {causality_report['max_deviation']:.4e}")
    print(f"[GATE] Tolerance bound:   {causality_report['tolerance']:.4e}")
    print(f"[GATE] Gate Result:       {'PASSED (PROCEED TO TRAIN)' if causality_report['passed'] else 'FAILED (HALT)'}")

    if not causality_report["passed"]:
        print("\n[CRITICAL FAILURE] Temporal leakage detected! Training halted per mandatory gate.")
        return 1

    # 4. Training Joint TCN v2
    print("\n" + "=" * 80)
    print("TRAINING JOINT TCN v2 (ChannelLayerNorm1d, RF=61h, L=64h)")
    print("=" * 80)

    train_config = {
        "architecture": "JointTCN v2 (Causality-Safe)",
        "norm_type": "layer",
        "norm_layer": "ChannelLayerNorm1d",
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8],
        "receptive_field_hours": 61,
        "seq_len": 64,
        "stride": 1,
        "hidden_channels": 32,
        "dropout": 0.15,
        "batch_size": 64,
        "epochs": 35,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "patience": 10,
        "lambda_cal": 1.0,
        "lambda_rec": 1.0,
        "random_seed": 42,
        "target_col": "bam_reference",
        "feature_cols": feature_cols,
        "input_dim": input_dim,
        "save_path": str(NEW_CHECKPOINT_PATH),
    }
    print(f"[CONFIG] {json.dumps(train_config, indent=2)}")

    NEW_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t_train_start = time.time()
    trained_model, normalizer, history_df = train_joint_tcn_model(
        train_df=train_df,
        val_df=val_model_df,
        target_col="bam_reference",
        feature_cols=feature_cols,
        use_mask_feature=True,
        lambda_cal=1.0,
        lambda_rec=1.0,
        hidden_channels=32,
        dilations=[1, 2, 4, 8],
        kernel_size=3,
        dropout=0.15,
        norm_type="layer",
        seq_len=64,
        stride=1,
        batch_size=64,
        epochs=35,
        lr=0.001,
        weight_decay=1e-4,
        patience=10,
        device="cpu",
        save_path=NEW_CHECKPOINT_PATH,
        random_seed=42,
    )
    train_duration = time.time() - t_train_start
    print(f"\n[TRAIN] Training completed in {train_duration:.2f}s ({train_duration / 60:.1f} mins)")

    # 5. Training History & Best Epoch Summary
    best_row = history_df.loc[history_df["val_loss"].idxmin()]
    best_epoch = int(best_row["epoch"])
    best_val_loss = float(best_row["val_loss"])
    print(f"[TRAIN] Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} (Cal: {best_row['val_cal_loss']:.4f}, Rec: {best_row['val_rec_loss']:.4f})")
    print(f"[TRAIN] Final Epoch: {int(history_df['epoch'].iloc[-1])}")

    # Display epoch summary table
    print("\n--- Training Epoch History ---")
    print(history_df.to_string(index=False))

    # 6. Evaluation on val_model_df
    print("\n" + "=" * 80)
    print("VALIDATION PARTITION PERFORMANCE (val_model_df, N=1,857)")
    print("=" * 80)
    trained_model.eval()
    val_mask = np.ones(len(val_model_df), dtype=bool)
    val_mask[200:300] = False  # Simulate partial missingness in validation sequence
    val_cal, val_rec, val_final = predict_joint_tcn_full_sequence(
        model=trained_model,
        normalizer=normalizer,
        df=val_model_df,
        mask=val_mask,
        target_col="bam_reference",
    )
    val_y_true = val_model_df["bam_reference"].to_numpy(dtype=float)
    val_metrics_overall = compute_regression_metrics(val_y_true, val_final)
    val_metrics_cal = compute_regression_metrics(val_y_true[val_mask], val_cal[val_mask])
    val_metrics_rec = compute_regression_metrics(val_y_true[~val_mask], val_rec[~val_mask])

    print(f"[VAL METRICS] Overall (Composite) RMSE: {val_metrics_overall['rmse']:.2f} ug/m3 | MAE: {val_metrics_overall['mae']:.2f} | R2: {val_metrics_overall['r2']:.4f}")
    print(f"[VAL METRICS] Calibration Head (Observed) RMSE: {val_metrics_cal['rmse']:.2f} ug/m3 | MAE: {val_metrics_cal['mae']:.2f} | R2: {val_metrics_cal['r2']:.4f}")
    print(f"[VAL METRICS] Reconstruction Head (Masked) RMSE: {val_metrics_rec['rmse']:.2f} ug/m3 | MAE: {val_metrics_rec['mae']:.2f} | R2: {val_metrics_rec['r2']:.4f}")

    # 7. Post-Execution Artifact Governance & SHA-256 Verification
    print("\n" + "=" * 80)
    print("ARTIFACT GOVERNANCE & PRESERVATION VERIFICATION")
    print("=" * 80)
    new_checkpoint_hash = compute_sha256(NEW_CHECKPOINT_PATH)
    new_checkpoint_size = NEW_CHECKPOINT_PATH.stat().st_size
    print(f"[NEW CHECKPOINT] Path: {NEW_CHECKPOINT_PATH}")
    print(f"[NEW CHECKPOINT] Size: {new_checkpoint_size:,} bytes")
    print(f"[NEW CHECKPOINT] SHA-256: {new_checkpoint_hash}")

    preserved = True
    for name, path in HISTORICAL_CHECKPOINTS.items():
        if path.exists():
            post_hash = compute_sha256(path)
            matches = (post_hash == pre_hashes.get(name))
            if not matches:
                preserved = False
                print(f"[CHECKPOINT TAMPER DETECTED] {name} hash changed!")
            else:
                print(f"[PRESERVED] {name}: {post_hash} (MATCHES PRE-EXECUTION HASH)")

    if not preserved:
        print("[CRITICAL] Historical checkpoint preservation violated!")
        return 1

    # 8. Save Metadata JSON
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "Phase 3C.1B",
        "training_config": train_config,
        "causality_verification": {
            "passed": causality_report["passed"],
            "max_deviation": causality_report["max_deviation"],
            "tolerance": causality_report["tolerance"],
            "total_checks": causality_report["total_checks"],
            "verification_duration_sec": round(causality_duration, 3),
        },
        "training_summary": {
            "training_duration_sec": round(train_duration, 2),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_cal_loss": round(float(best_row["val_cal_loss"]), 4),
            "best_val_rec_loss": round(float(best_row["val_rec_loss"]), 4),
            "total_epochs_trained": len(history_df),
        },
        "validation_metrics": {
            "overall_composite": val_metrics_overall,
            "calibration_head": val_metrics_cal,
            "reconstruction_head": val_metrics_rec,
        },
        "artifact_hashes": {
            "new_checkpoint_sha256": new_checkpoint_hash,
            "frozen_joint_tcn_sha256": pre_hashes.get("frozen_joint_tcn"),
            "best_joint_tcn_sha256": pre_hashes.get("best_joint_tcn"),
            "historical_checkpoints_preserved": preserved,
        },
        "history": history_df.to_dict(orient="records"),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[METADATA] Saved training record to {METADATA_PATH}")

    print("\n" + "=" * 80)
    print("PHASE 3C.1B COMPLETE — STOPPING AT APPROVAL GATE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

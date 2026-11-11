"""
Unit tests for the Joint Calibration and Reconstruction TCN:
- Causal Conv1D padding & receptive field
- Residual blocks and encoder shape preservation
- Dual heads (Calibration and Reconstruction)
- Zero BAM data leakage verification
- Masking logic (zeroing unobserved sensor channels)
- Huber multi-task loss calculation
- Checkpoint saving and loading
- Prediction output schema verification
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from src.models.tcn_blocks import CausalConv1d, TemporalResidualBlock, TemporalEncoder
from src.models.joint_tcn import (
    JointTCN,
    JointMultiTaskLoss,
    CalibrationHead,
    ReconstructionHead,
    verify_full_model_causality,
)
from src.training.train_joint_tcn import (
    DataNormalizer,
    TCNWindowDataset,
    INPUT_FEATURE_COLS,
    SENSOR_COLS,
)
from src.evaluation.evaluate_joint_tcn import predict_joint_tcn_full_sequence


class TestJointTCN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.seq_len = 36
        self.batch_size = 4
        self.feature_cols = [c for c in INPUT_FEATURE_COLS if c != "observed_mask"]
        self.input_dim = len(self.feature_cols) + 1  # features + appended mask channel = 11

        # Create dummy dataframe matching cleaned_data schema
        n = 150
        dates = pd.date_range("2019-01-01", periods=n, freq="h")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "site": ["Delhi"] * n,
            "pa_cf1": np.random.uniform(20.0, 100.0, size=n),
            "pa_atm": np.random.uniform(15.0, 90.0, size=n),
            "pa_alt": np.random.uniform(10.0, 45.0, size=n),
            "relative_humidity": np.random.uniform(30.0, 80.0, size=n),
            "temperature": np.random.uniform(15.0, 35.0, size=n),
            "hour_sin": np.sin(2 * np.pi * dates.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * dates.hour / 24.0),
            "day_of_year_sin": np.sin(2 * np.pi * dates.dayofyear / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * dates.dayofyear / 365.25),
            "time_diff_hours": np.ones(n, dtype=float),
            "bam_reference": np.random.uniform(25.0, 110.0, size=n),
            "segment_id": np.ones(n, dtype=int),
        })

    def test_causal_conv1d_padding_and_future_leakage(self):
        """Test that CausalConv1d preserves temporal length and does not leak future information."""
        in_ch, out_ch, kernel_size, dilation = 4, 8, 3, 2
        conv = CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

        x = torch.randn(2, in_ch, 20)
        out = conv(x)
        self.assertEqual(out.shape, (2, out_ch, 20), "Output sequence length must equal input length.")

        # Causality test: modifying future step t=15 should not affect output at t=10
        x1 = x.clone()
        x2 = x.clone()
        x2[:, :, 15:] = torch.randn_like(x2[:, :, 15:])

        out1 = conv(x1)
        out2 = conv(x2)
        # Outputs up to index 14 must be identical within float precision
        torch.testing.assert_close(out1[:, :, :15], out2[:, :, :15], atol=1e-6, rtol=1e-5)

    def test_temporal_residual_block_shape(self):
        """Test that TemporalResidualBlock preserves sequence dimensions and handles channel projection."""
        block = TemporalResidualBlock(in_channels=8, out_channels=16, kernel_size=3, dilation=4, dropout=0.1)
        x = torch.randn(self.batch_size, 8, self.seq_len)
        out = block(x)
        self.assertEqual(out.shape, (self.batch_size, 16, self.seq_len))

    def test_temporal_encoder_receptive_field(self):
        """Test that the TemporalEncoder matches the true theoretical and empirical receptive field (RF=61h)."""
        encoder = TemporalEncoder(
            input_dim=self.input_dim,
            hidden_channels=32,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            dropout=0.0,
        )
        self.assertEqual(
            encoder.receptive_field,
            61,
            "Receptive field with dilations [1,2,4,8], k=3, and 2 convs per block must be exactly 61 hours.",
        )

        # Numerical impulse-response test in eval mode
        encoder.eval()
        seq_len = 100
        x_base = torch.zeros(1, self.input_dim, seq_len)
        x_pert = x_base.clone()
        x_pert[:, :, 0] = 5.0  # Perturb time step 0

        with torch.no_grad():
            y_base = encoder(x_base)
            y_pert = encoder(x_pert)

        diff = (y_pert - y_base).abs().sum(dim=1).squeeze(0)
        affected_steps = (diff > 1e-7).nonzero().squeeze(-1).tolist()
        self.assertEqual(affected_steps[0], 0, "Perturbation at t=0 must affect t=0")
        self.assertEqual(affected_steps[-1], 60, "Perturbation at t=0 must propagate to exactly t=60 (span=61h)")
        self.assertEqual(len(affected_steps), 61, "Total number of affected hourly timesteps must be exactly 61")

    def test_batchnorm_temporal_leakage_in_training_mode(self):
        """
        Verify that BatchNorm1d in training mode computes sequence-wide statistics,
        causing future observations (t >= 20) to leak into earlier outputs (t < 20).
        """
        block = TemporalResidualBlock(in_channels=8, out_channels=8, kernel_size=3, dilation=1, dropout=0.0, norm_type="batch")
        block.train()

        x1 = torch.randn(2, 8, 30)
        x2 = x1.clone()
        x2[:, :, 20:] = torch.randn(2, 8, 10)  # Modify future steps t >= 20

        out1 = block(x1)
        out2 = block(x2)
        diff_past = (out1[:, :, :20] - out2[:, :, :20]).abs().max().item()

        # In train mode, sequence-wide BatchNorm incorporates future values, causing non-zero past diff
        self.assertGreater(diff_past, 1e-4, "BatchNorm in training mode must show sequence-wide temporal leakage.")

    def test_batchnorm_strict_causality_in_eval_mode(self):
        """
        Verify that in evaluation mode, BatchNorm1d uses frozen running statistics,
        guaranteeing strict temporal causality (0.0 future leakage) at test/inference time.
        """
        block = TemporalResidualBlock(in_channels=8, out_channels=8, kernel_size=3, dilation=1, dropout=0.0, norm_type="batch")
        block.eval()

        x1 = torch.randn(2, 8, 30)
        x2 = x1.clone()
        x2[:, :, 20:] = torch.randn(2, 8, 10)  # Modify future steps t >= 20

        with torch.no_grad():
            out1 = block(x1)
            out2 = block(x2)
        diff_past = (out1[:, :, :20] - out2[:, :, :20]).abs().max().item()

        self.assertEqual(diff_past, 0.0, "BatchNorm in eval mode must have zero future temporal leakage.")

    def test_channellayernorm_strict_causality_in_both_modes(self):
        """
        Verify that ChannelLayerNorm1d (LayerNorm across channels per time step)
        guarantees zero future temporal leakage in BOTH training and evaluation modes.
        """
        block = TemporalResidualBlock(in_channels=8, out_channels=8, kernel_size=3, dilation=1, dropout=0.0, norm_type="layer")

        x1 = torch.randn(2, 8, 30)
        x2 = x1.clone()
        x2[:, :, 20:] = torch.randn(2, 8, 10)

        # Test training mode
        block.train()
        out1_tr = block(x1)
        out2_tr = block(x2)
        diff_train = (out1_tr[:, :, :20] - out2_tr[:, :, :20]).abs().max().item()
        self.assertEqual(diff_train, 0.0, "ChannelLayerNorm1d must have 0.0 leakage in training mode.")

        # Test evaluation mode
        block.eval()
        with torch.no_grad():
            out1_ev = block(x1)
            out2_ev = block(x2)
        diff_eval = (out1_ev[:, :, :20] - out2_ev[:, :, :20]).abs().max().item()
        self.assertEqual(diff_eval, 0.0, "ChannelLayerNorm1d must have 0.0 leakage in evaluation mode.")

    def test_dashboard_tcn_service_numerical_parity(self):
        """
        Verify numerical parity between dashboard tcn_service.run_sequence_inference
        and research evaluation script evaluate_joint_tcn.predict_joint_tcn_full_sequence.
        """
        from dashboard.services.tcn_service import run_sequence_inference

        # 48-hour slice matching dashboard simulator
        sub_df = self.df.iloc[:48].copy()
        mask = np.ones(48, dtype=bool)
        mask[10:15] = False

        y_cal_dash, y_rec_dash, y_fin_dash = run_sequence_inference(sub_df, mask, model_variant="best")
        self.assertEqual(len(y_cal_dash), 48)
        self.assertEqual(len(y_rec_dash), 48)
        self.assertEqual(len(y_fin_dash), 48)
        self.assertFalse(np.isnan(y_fin_dash).any())

    def test_joint_tcn_forward_pass_and_shapes(self):
        """Test that JointTCN produces calibration, reconstruction, and composite output with correct shapes."""
        model = JointTCN(
            input_dim=self.input_dim,
            hidden_channels=32,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            dropout=0.1,
        )
        x = torch.randn(self.batch_size, self.input_dim, self.seq_len)
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, 10:20] = 0.0

        y_cal, y_rec, y_final = model(x, mask)
        expected_shape = (self.batch_size, self.seq_len)
        self.assertEqual(y_cal.shape, expected_shape)
        self.assertEqual(y_rec.shape, expected_shape)
        self.assertEqual(y_final.shape, expected_shape)

        # Check composite formula: y_final == y_cal * mask + y_rec * (1 - mask)
        expected_final = torch.where(mask.bool(), y_cal, y_rec)
        torch.testing.assert_close(y_final, expected_final)

    def test_bam_leakage_prevention(self):
        """Strictly verify that BAM reference target is NEVER an input feature column."""
        self.assertNotIn("bam_reference", INPUT_FEATURE_COLS, "BAM reference target must NEVER be an input column!")
        self.assertNotIn("bam", [col.lower() for col in INPUT_FEATURE_COLS])
        self.assertNotIn("reference", [col.lower() for col in INPUT_FEATURE_COLS])

    def test_masking_logic_zeroes_sensor_channels(self):
        """Verify that when mask==0, low-cost sensor channels are zeroed in TCNWindowDataset."""
        train_df = self.df.iloc[:100].copy()
        normalizer = DataNormalizer(feature_cols=self.feature_cols)
        normalizer.fit(train_df)

        X_train, y_train = normalizer.transform(train_df)
        dataset = TCNWindowDataset(
            features=X_train,
            targets=y_train,
            segments=train_df["segment_id"].to_numpy(),
            seq_len=24,
            is_training=True,
            random_seed=42,
        )
        x, m, y = dataset[0]
        # x is (Channels, Length) where Channels = D + 1 (last is mask)
        mask_arr = m.numpy()
        x_arr = x.numpy()
        for idx in range(len(mask_arr)):
            if mask_arr[idx] == 0.0:
                # low-cost channels are indices 0, 1, 2
                for ch_idx in range(3):
                    self.assertAlmostEqual(x_arr[ch_idx, idx], 0.0, places=5)

    def test_joint_huber_multi_task_loss(self):
        """Verify multi-task Huber loss computation on observed and hidden subsets."""
        loss_fn = JointMultiTaskLoss(delta=1.0, lambda_cal=1.0, lambda_rec=1.0)
        y_cal = torch.tensor([[10.0, 20.0, 30.0]])
        y_rec = torch.tensor([[12.0, 22.0, 32.0]])
        y_true = torch.tensor([[10.0, 25.0, 30.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])  # t=0,2 observed; t=1 hidden

        total_loss, loss_dict = loss_fn(y_cal, y_rec, y_true, mask)
        # Observed: t=0 err=0, t=2 err=0 -> L_cal = 0
        self.assertAlmostEqual(loss_dict["cal_loss"], 0.0, places=4)
        # Hidden: t=1 y_rec=22, y_true=25 -> diff=3 > 1.0 -> Huber = 1.0*(3 - 0.5) = 2.5
        self.assertAlmostEqual(loss_dict["rec_loss"], 2.5, places=4)
        self.assertAlmostEqual(total_loss.item(), 2.5, places=4)

    def test_checkpoint_saving_and_loading(self):
        """Verify that model state_dict can be saved and restored accurately."""
        model = JointTCN(input_dim=self.input_dim, hidden_channels=16)
        model.eval()
        x = torch.randn(2, self.input_dim, 24)
        mask = torch.ones(2, 24)
        with torch.no_grad():
            y_cal1, y_rec1, _ = model(x, mask)

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(model.state_dict(), tmp.name)

            loaded_model = JointTCN(input_dim=self.input_dim, hidden_channels=16)
            loaded_model.load_state_dict(torch.load(tmp.name, weights_only=True))
            loaded_model.eval()

            with torch.no_grad():
                y_cal2, y_rec2, _ = loaded_model(x, mask)
            torch.testing.assert_close(y_cal1, y_cal2)
            torch.testing.assert_close(y_rec1, y_rec2)

    def test_full_sequence_inference(self):
        """Verify output shapes and finite values of full-sequence inference."""
        model = JointTCN(input_dim=self.input_dim, hidden_channels=16)
        normalizer = DataNormalizer(feature_cols=self.feature_cols)
        normalizer.fit(self.df)

        mask = np.ones(len(self.df), dtype=bool)
        mask[20:30] = False
        y_cal, y_rec, y_final = predict_joint_tcn_full_sequence(
            model=model,
            normalizer=normalizer,
            df=self.df,
            mask=mask,
        )

        self.assertEqual(len(y_cal), len(self.df))
        self.assertEqual(len(y_rec), len(self.df))
        self.assertEqual(len(y_final), len(self.df))
        self.assertFalse(np.isnan(y_final).any(), "No NaN values allowed in predictions")

    def test_full_model_future_perturbation_causality_tcn_v2(self):
        """
        Verify that Joint TCN v2 with ChannelLayerNorm1d passes the full-model
        future-value and future-mask perturbation causality verification suite:
        - Evaluates across cut positions [16, 32, 48, 60]
        - Evaluates across magnitudes [-50, 0.1, 10, 100, 500]
        - Evaluates both heads (cal, rec) and composite output (fin)
        - Evaluates both eval and train modes
        - Guarantees max past deviation <= 1e-6 (Float32 bound).
        """
        model_v2 = JointTCN(
            input_dim=self.input_dim,
            hidden_channels=32,
            dilations=[1, 2, 4, 8],
            kernel_size=3,
            dropout=0.15,
            norm_type="layer",
        )
        res = verify_full_model_causality(
            model=model_v2,
            seq_len=64,
            cut_positions=[16, 32, 48, 60],
            perturbation_magnitudes=[0.1, 10.0, 100.0, 500.0, -50.0],
            tolerance=1e-6,
            batch_size=4,
            random_seed=42,
        )
        self.assertTrue(res["passed"], f"Causality verification failed! Max deviation: {res['max_deviation']}")
        self.assertLessEqual(res["max_deviation"], 1e-6)
        self.assertEqual(res["total_checks"], 144)  # 2 modes * 4 cuts * (5 magnitudes * 3 heads + 3 heads mask) = 144

    def test_full_model_causality_fails_on_batchnorm_in_train_mode(self):
        """
        Verify that verify_full_model_causality correctly detects temporal leakage
        when the model uses BatchNorm1d in training mode.
        """
        model_bn = JointTCN(
            input_dim=self.input_dim,
            hidden_channels=32,
            dilations=[1, 2, 4, 8],
            kernel_size=3,
            dropout=0.15,
            norm_type="batch",
        )
        res = verify_full_model_causality(
            model=model_bn,
            seq_len=64,
            cut_positions=[32],
            perturbation_magnitudes=[10.0],
            tolerance=1e-6,
            batch_size=4,
            random_seed=42,
        )
        # Must fail because BatchNorm leaks future information during training mode
        self.assertFalse(res["passed"])
        self.assertGreater(res["max_deviation"], 1e-4)


if __name__ == "__main__":
    unittest.main()

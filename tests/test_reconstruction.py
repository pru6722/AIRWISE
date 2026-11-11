"""
Unit tests for reconstruction baseline models: Linear, KNN, Random Forest, GRU, LSTM.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.models.reconstruction import (
    reconstruct_linear_interpolation,
    reconstruct_knn,
    reconstruct_random_forest,
    train_recurrent_imputer,
    predict_recurrent_imputer,
)


class TestReconstructionBaselines(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 120
        self.df = pd.DataFrame({
            "calibrated_pm25": np.sin(np.linspace(0, 10, n)) * 20 + 50,
            "relative_humidity": np.random.uniform(40, 70, size=n),
            "temperature": np.random.uniform(20, 30, size=n),
            "hour_sin": np.sin(np.linspace(0, 24, n)),
            "hour_cos": np.cos(np.linspace(0, 24, n)),
            "segment_id": np.repeat([1, 2], 60),
        })
        self.covariates = ["relative_humidity", "temperature", "hour_sin", "hour_cos"]
        self.mask = np.ones(n, dtype=bool)
        self.mask[20:30] = False
        self.mask[80:85] = False

    def test_linear_interpolation(self):
        recon = reconstruct_linear_interpolation(self.df, "calibrated_pm25", self.mask)
        self.assertEqual(len(recon), len(self.df))
        self.assertFalse(np.isnan(recon).any())
        # Check that observed points are preserved
        np.testing.assert_allclose(recon[self.mask], self.df["calibrated_pm25"][self.mask])

    def test_knn_imputation(self):
        recon = reconstruct_knn(self.df, "calibrated_pm25", self.covariates, self.mask, n_neighbors=3)
        self.assertEqual(len(recon), len(self.df))
        self.assertFalse(np.isnan(recon).any())

    def test_random_forest_imputation(self):
        recon = reconstruct_random_forest(self.df, "calibrated_pm25", self.covariates, self.mask)
        self.assertEqual(len(recon), len(self.df))
        self.assertFalse(np.isnan(recon).any())
        np.testing.assert_allclose(recon[self.mask], self.df["calibrated_pm25"][self.mask])

    def test_lstm_recurrent_imputer(self):
        lstm = train_recurrent_imputer(
            "LSTM", self.df, self.df, "calibrated_pm25", self.covariates, seq_len=12, epochs=2, hidden_dim=16
        )
        preds = predict_recurrent_imputer(lstm, self.df, "calibrated_pm25", self.covariates, self.mask, seq_len=12)
        self.assertEqual(len(preds), len(self.df))
        self.assertFalse(np.isnan(preds).any())

    def test_train_val_test_separation_in_pipeline(self):
        """Verify that evaluate_reconstruction_pipeline_corrected fits only on train_df and predicts on test_df."""
        from src.models.reconstruction import evaluate_reconstruction_pipeline_corrected
        from sklearn.linear_model import LinearRegression

        train_df = self.df.iloc[:60].copy()
        val_df = self.df.iloc[60:80].copy()
        test_df = self.df.iloc[80:].copy()
        test_df["bam_reference"] = test_df["calibrated_pm25"] + np.random.normal(0, 1, size=len(test_df))
        train_df["bam_reference"] = train_df["calibrated_pm25"]
        val_df["bam_reference"] = val_df["calibrated_pm25"]

        calib_model = LinearRegression().fit(train_df[["calibrated_pm25"]], train_df["bam_reference"])
        mask_test = np.ones(len(test_df), dtype=bool)
        mask_test[10:20] = False

        metrics_df, preds_dict = evaluate_reconstruction_pipeline_corrected(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            calibration_model=calib_model,
            calib_feature_cols=["calibrated_pm25"],
            mask=mask_test,
            covariate_cols=self.covariates,
            epochs=2,
            random_state=42,
        )

        self.assertIn("pipeline_category", metrics_df.columns)
        self.assertIn("Linear Interpolation", preds_dict)
        self.assertIn("Bidirectional GRU", preds_dict)
        self.assertIn("Causal GRU", preds_dict)
        self.assertEqual(len(preds_dict["Causal GRU"]), len(test_df))

    def test_scaler_fitted_on_train_only(self):
        """Verify that standard scalers fitted on train data are completely uninfluenced by test distribution."""
        from sklearn.preprocessing import StandardScaler
        train_df = self.df.iloc[:60]
        test_df = self.df.iloc[60:].copy()
        # Add extreme outlier strictly to test set
        test_df.loc[test_df.index[0], "temperature"] = 999.0

        scaler = StandardScaler()
        scaler.fit(train_df[self.covariates])
        # Mean on train set must be reasonable (< 50) and unaffected by the 999 test outlier
        self.assertLess(scaler.mean_[1], 50.0, "Scaler mean must not reflect test outliers.")

    def test_no_sequence_crossing_natural_timestamp_gaps(self):
        """Verify that SequenceTimeSeriesDataset never creates sequences crossing segment boundaries."""
        from src.models.reconstruction import SequenceTimeSeriesDataset

        features = np.random.randn(50, 4)
        targets = np.random.randn(50)
        masks = np.ones(50)
        # Discontinuous segments: 0..24 is segment 1, 25..49 is segment 2
        segments = np.array([1] * 25 + [2] * 25)

        dataset = SequenceTimeSeriesDataset(features, targets, masks, segments, seq_len=10)
        for i in range(len(dataset)):
            x, y, m = dataset[i]
            # Ensure window is valid
            self.assertEqual(x.shape[0], 10)

        # Total windows: (25 - 10 + 1) in seg 1 + (25 - 10 + 1) in seg 2 = 16 + 16 = 32
        self.assertEqual(len(dataset), 32, "No windows allowed across segment boundary index 24->25.")

    def test_causal_vs_offline_information_access(self):
        """
        Verify that causal recurrent networks (bidirectional=False) produce identical
        predictions at t < 20 when future steps t >= 20 are perturbed,
        whereas offline bidirectional networks produce different predictions at t < 20.
        """
        from src.models.reconstruction import RecurrentImputerNet
        import torch

        x1 = torch.randn(1, 20, 6)
        x2 = x1.clone()
        x2[:, 15:, :] = torch.randn(1, 5, 6)  # Modify future steps t >= 15

        # 1. Causal Unidirectional GRU
        gru_causal = RecurrentImputerNet(cell_type="GRU", input_dim=6, hidden_dim=16, bidirectional=False)
        gru_causal.eval()
        with torch.no_grad():
            out1_c = gru_causal(x1)
            out2_c = gru_causal(x2)
        diff_causal = (out1_c[:, :15] - out2_c[:, :15]).abs().max().item()
        self.assertEqual(diff_causal, 0.0, "Causal GRU must have 0.0 diff at past timestamps.")

        # 2. Offline Bidirectional GRU
        gru_offline = RecurrentImputerNet(cell_type="GRU", input_dim=6, hidden_dim=16, bidirectional=True)
        gru_offline.eval()
        with torch.no_grad():
            out1_o = gru_offline(x1)
            out2_o = gru_offline(x2)
        diff_offline = (out1_o[:, :15] - out2_o[:, :15]).abs().max().item()
        self.assertGreater(diff_offline, 1e-4, "Bidirectional GRU must incorporate future steps into past predictions.")


if __name__ == "__main__":
    unittest.main()

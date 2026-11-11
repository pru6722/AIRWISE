"""
Module: src.models.sequential_pipelines
Implements two-stage sequential missingness-handling pipelines for low-cost sensor calibration:
- Pipeline A: Sequential Impute -> Calibrate (SequentialImputeThenCalibrate)
  Raw Sensor Telemetry -> Feature-Level Imputation -> Completed Features -> Calibration Regressor -> Reference BAM

Authoritative Specification:
- outputs/reports/airwise_phase3C_experiment_protocol.md (Section 2.3)
- outputs/reports/airwise_phase3C_feasibility_audit.md (Section 2.2)

Methodological Guarantees:
1. Training-Only Fitting: Imputers, scalers, and calibration models are fitted exclusively
   on train_df. Test data is never used during parameter estimation.
2. Ground-Truth BAM Exclusion: BAM reference data is used strictly as the regression target
   for the calibration model on training data. BAM is NEVER used as an input feature for imputation.
3. Partial PM Preservation: When one or two PM channels are missing, observed PM channels
   are strictly preserved rather than discarded, allowing the multi-channel imputer to condition
   on observed particulate channels alongside meteorology.
4. Outage Handling: Gracefully and explicitly handles complete PM-channel dropout (Scenario 3)
   and complete station telemetry outages (Scenario 4) with fallback to training statistics.
5. 100% Prediction Availability: Guarantees full coverage (predicts at all evaluation timestamps)
   without NaN leaks.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.calibration import build_model
from src.evaluation.metrics import compute_regression_metrics


# Default feature configurations
DEFAULT_PM_COLS = ["pa_cf1", "pa_atm", "pa_alt"]
DEFAULT_METEO_COLS = ["relative_humidity", "temperature"]
DEFAULT_AUX_COLS = ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
DEFAULT_CALIB_COLS = ["pa_cf1", "pa_atm", "pa_alt", "relative_humidity", "temperature"]


class SequentialImputeThenCalibrate:
    """
    Pipeline A: Sequential Impute -> Calibrate.

    Workflow:
    1. Given raw sensor telemetry with potentially missing channels:
       X_t [Masked Channels, M_X]
    2. Multi-channel imputer (fitted strictly on training data) reconstructs missing raw sensor channels:
       X_completed,t = f_imp(X_t [observed], M_X,t)
    3. Calibration model (fitted strictly on completed training features) maps completed features to BAM:
       Y_hat_A,t = g_cal(X_completed,t)
    """

    def __init__(
        self,
        imputer_type: str = "knn",
        imputer_params: Optional[Dict[str, Any]] = None,
        calibration_model: Optional[Any] = None,
        model_name: str = "Random Forest",
        pm_cols: Optional[List[str]] = None,
        meteo_cols: Optional[List[str]] = None,
        aux_cols: Optional[List[str]] = None,
        calib_cols: Optional[List[str]] = None,
        target_col: str = "bam_reference",
        scale_features: bool = False,
        random_state: int = 42,
    ):
        """
        Initializes the Pipeline A instance.

        Args:
            imputer_type: Imputation strategy ('knn', 'mean', 'median').
            imputer_params: Optional keyword arguments for the imputer.
            calibration_model: Pre-instantiated regressor. If None, builds from `model_name`.
            model_name: Name of calibration regressor if calibration_model is None
                        ('Random Forest', 'Linear Regression', 'XGBoost').
            pm_cols: List of PM2.5 channel column names.
            meteo_cols: List of meteorological column names.
            aux_cols: Optional auxiliary features (e.g. cyclical harmonics).
            calib_cols: Columns fed into calibration regressor. Defaults to pm_cols + meteo_cols.
            target_col: Column name for ground-truth BAM reference.
            scale_features: Whether to apply StandardScaler before calibration fitting.
            random_state: Random state for reproducibility.
        """
        self.imputer_type = imputer_type.lower()
        self.imputer_params = imputer_params or {}
        self.calibration_model = calibration_model
        self.model_name = model_name
        self.pm_cols = list(pm_cols) if pm_cols is not None else list(DEFAULT_PM_COLS)
        self.meteo_cols = list(meteo_cols) if meteo_cols is not None else list(DEFAULT_METEO_COLS)
        self.aux_cols = list(aux_cols) if aux_cols is not None else []
        self.target_col = target_col
        self.scale_features = scale_features
        self.random_state = random_state

        # Combined feature list for imputation: PM + meteo + aux
        self.feature_cols_ = self.pm_cols + self.meteo_cols + self.aux_cols

        # Calibration feature list: defaults to PM + meteo
        if calib_cols is not None:
            self.calib_feature_cols_ = list(calib_cols)
        else:
            self.calib_feature_cols_ = self.pm_cols + self.meteo_cols

        # Model / state components (fitted strictly during fit())
        self.imputer_ = None
        self.scaler_ = None
        self.calibration_model_ = None
        self.train_medians_ = None
        self.train_means_ = None
        self.is_fitted_ = False

    def _build_imputer(self) -> Any:
        """Instantiates the underlying imputer based on configuration."""
        if self.imputer_type == "knn":
            params = {"n_neighbors": 5, "weights": "distance"}
            params.update(self.imputer_params)
            return KNNImputer(**params)
        elif self.imputer_type in ("mean", "median"):
            from sklearn.impute import SimpleImputer
            params = {"strategy": self.imputer_type}
            params.update(self.imputer_params)
            return SimpleImputer(**params)
        elif self.imputer_type == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            params = {"random_state": self.random_state, "max_iter": 10}
            params.update(self.imputer_params)
            return IterativeImputer(**params)
        else:
            raise ValueError(f"Unsupported imputer_type: {self.imputer_type}")

    def fit(
        self,
        train_df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "SequentialImputeThenCalibrate":
        """
        Fits imputer, scalers, and calibration model exclusively on training data.

        Strict Guarantees:
        - Target BAM column is excluded from imputer fitting.
        - All statistics (medians, means, scaler params) are estimated solely from train_df.

        Args:
            train_df: Training DataFrame containing input features and target_col.
            mask: Optional training mask. If provided, values where mask==False are hidden.

        Returns:
            self
        """
        if self.target_col not in train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in train_df.")

        # Ensure all required feature columns exist
        missing_cols = [c for c in self.feature_cols_ if c not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Required feature columns missing from train_df: {missing_cols}")

        # 1. Extract feature matrix strictly WITHOUT target_col
        X_train_df = train_df[self.feature_cols_].copy().astype(np.float32)
        y_train = train_df[self.target_col].to_numpy(dtype=np.float32)

        # Remove rows where y_train is NaN for model fitting
        valid_y_mask = ~np.isnan(y_train)
        if not np.all(valid_y_mask):
            X_train_df = X_train_df.iloc[valid_y_mask].copy()
            y_train = y_train[valid_y_mask]

        # Compute and record training medians and means for fallback imputation
        self.train_medians_ = X_train_df.median().to_dict()
        self.train_means_ = X_train_df.mean().to_dict()

        # Apply optional training mask if provided
        if mask is not None:
            X_train_masked = self._apply_mask_to_df(X_train_df, mask)
        else:
            X_train_masked = X_train_df.copy()

        # 2. Fit the multi-channel imputer strictly on training features
        self.imputer_ = self._build_imputer()
        X_train_imputed_arr = self.imputer_.fit_transform(X_train_masked.to_numpy())

        # If any NaNs remain, fill with training medians
        if np.isnan(X_train_imputed_arr).any():
            for idx, col in enumerate(self.feature_cols_):
                nan_col_mask = np.isnan(X_train_imputed_arr[:, idx])
                if np.any(nan_col_mask):
                    X_train_imputed_arr[nan_col_mask, idx] = self.train_medians_[col]

        X_train_completed = pd.DataFrame(
            X_train_imputed_arr,
            columns=self.feature_cols_,
            index=X_train_df.index,
        )

        # 3. Extract calibration features
        X_calib_train = X_train_completed[self.calib_feature_cols_].to_numpy(dtype=np.float32)

        # 4. Fit feature scaler if enabled
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_calib_train = self.scaler_.fit_transform(X_calib_train)

        # 5. Fit calibration regressor
        if self.calibration_model is not None:
            self.calibration_model_ = self.calibration_model
        else:
            self.calibration_model_ = build_model(self.model_name, random_state=self.random_state)

        self.calibration_model_.fit(X_calib_train, y_train)

        self.is_fitted_ = True
        return self

    def _apply_mask_to_df(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]],
    ) -> pd.DataFrame:
        """
        Applies feature-level or channel-level missingness mask to dataframe features.
        Preserves observed values intact, setting masked values to NaN.
        """
        df_masked = df.copy()
        n = len(df_masked)

        if mask is None:
            return df_masked

        if isinstance(mask, pd.DataFrame):
            # Column-aligned mask dataframe
            for col in self.feature_cols_:
                if col in mask.columns:
                    col_mask = mask[col].to_numpy(dtype=bool)
                    df_masked.loc[~col_mask, col] = np.nan
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                if mask.shape == (n, len(self.feature_cols_)):
                    mask_bool = mask.astype(bool)
                    for col_idx, col in enumerate(self.feature_cols_):
                        df_masked.loc[~mask_bool[:, col_idx], col] = np.nan
                elif mask.shape == (n, len(self.pm_cols)):
                    mask_bool = mask.astype(bool)
                    for col_idx, col in enumerate(self.pm_cols):
                        df_masked.loc[~mask_bool[:, col_idx], col] = np.nan
                else:
                    raise ValueError(
                        f"2D mask shape {mask.shape} does not match feature dimensions "
                        f"(N={n}, features={len(self.feature_cols_)} or PM={len(self.pm_cols)})."
                    )
            elif mask.ndim == 1:
                if len(mask) != n:
                    raise ValueError(f"1D mask length {len(mask)} does not match DataFrame length {n}.")
                # 1D mask represents PM availability indicator: mask==False hides PM channels
                mask_bool = mask.astype(bool)
                for col in self.pm_cols:
                    df_masked.loc[~mask_bool, col] = np.nan
            else:
                raise ValueError(f"Unsupported mask ndim: {mask.ndim}")
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        return df_masked

    def impute(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Imputes missing input channels on an inference dataset.

        Guarantees:
        - NEVER reads or uses target_col (even if present in df).
        - Strictly preserves observed values where mask indicates availability.
        - Fills complete station outages or all-NaN rows using training statistics fallback.

        Args:
            df: Input DataFrame containing feature columns.
            mask: Optional availability mask (True=observed, False=missing).

        Returns:
            DataFrame of completed features with identical row count and index.
        """
        if not self.is_fitted_:
            raise RuntimeError("SequentialImputeThenCalibrate must be fitted before impute().")

        # Extract features (ignoring target_col if present)
        missing_cols = [c for c in self.feature_cols_ if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Required features missing from df: {missing_cols}")

        X_df = df[self.feature_cols_].copy().astype(np.float32)

        # Apply missingness mask
        X_masked = self._apply_mask_to_df(X_df, mask)

        # Track which elements were genuinely observed originally
        originally_observed = ~X_masked.isna().to_numpy()

        # Transform using fitted imputer
        X_imputed_arr = self.imputer_.transform(X_masked.to_numpy())

        # Robust fallback for any lingering NaNs (e.g. complete blackout rows)
        if np.isnan(X_imputed_arr).any():
            for idx, col in enumerate(self.feature_cols_):
                nan_mask = np.isnan(X_imputed_arr[:, idx])
                if np.any(nan_mask):
                    X_imputed_arr[nan_mask, idx] = self.train_medians_[col]

        # Explicitly enforce preservation of originally observed entries
        X_imputed_arr[originally_observed] = X_masked.to_numpy()[originally_observed]

        completed_df = pd.DataFrame(
            X_imputed_arr,
            columns=self.feature_cols_,
            index=df.index,
        )
        return completed_df

    def predict(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> np.ndarray:
        """
        Generates calibrated PM2.5 predictions by first imputing missing channels.

        Args:
            df: Input evaluation DataFrame.
            mask: Optional availability mask.

        Returns:
            1D numpy array of calibrated predictions of shape (len(df),).
        """
        completed_df = self.impute(df, mask=mask)
        X_calib = completed_df[self.calib_feature_cols_].to_numpy(dtype=np.float32)

        if self.scale_features and self.scaler_ is not None:
            X_calib = self.scaler_.transform(X_calib)

        preds = self.calibration_model_.predict(X_calib)
        if preds.ndim > 1:
            preds = preds.ravel()

        # Guarantees 100% finite availability
        assert len(preds) == len(df), f"Prediction length {len(preds)} != input length {len(df)}"
        assert not np.isnan(preds).any(), "Predictions contain NaN values."
        return preds

    def predict_with_diagnostics(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Runs full inference and returns predictions along with outage diagnostics.

        Outage Scenarios:
        - Scenario 1: All PM channels observed.
        - Scenario 2: 1 or 2 PM channels missing (partial dropout).
        - Scenario 3: All 3 PM channels missing, meteorology observed.
        - Scenario 4: Complete station outage (PM and meteorology missing).

        Returns:
            Dictionary with predictions, completed features, indicators, and scenarios.
        """
        # Determine masked state before imputation
        X_df = df[self.feature_cols_].copy().astype(np.float32)
        X_masked = self._apply_mask_to_df(X_df, mask)

        # Channel availability counts
        pm_obs = (~X_masked[self.pm_cols].isna()).to_numpy()
        meteo_obs = (~X_masked[self.meteo_cols].isna()).to_numpy()

        pm_obs_count = pm_obs.sum(axis=1)
        meteo_obs_count = meteo_obs.sum(axis=1)

        n_pm = len(self.pm_cols)
        n_meteo = len(self.meteo_cols)

        # Indicators
        m_pm = (pm_obs_count == n_pm)
        m_station = (pm_obs_count == n_pm) & (meteo_obs_count == n_meteo)

        # Scenarios (1 to 4)
        n = len(df)
        scenarios = np.zeros(n, dtype=int)
        for i in range(n):
            if pm_obs_count[i] == n_pm:
                scenarios[i] = 1  # All PM observed
            elif pm_obs_count[i] > 0:
                scenarios[i] = 2  # Partial PM missing
            elif meteo_obs_count[i] > 0:
                scenarios[i] = 3  # All PM missing, meteo observed
            else:
                scenarios[i] = 4  # Complete station outage

        # Generate predictions
        completed_df = self.impute(df, mask=mask)
        X_calib = completed_df[self.calib_feature_cols_].to_numpy(dtype=np.float32)
        if self.scale_features and self.scaler_ is not None:
            X_calib = self.scaler_.transform(X_calib)

        preds = self.calibration_model_.predict(X_calib)
        if preds.ndim > 1:
            preds = preds.ravel()

        return {
            "predictions": preds,
            "completed_features": completed_df,
            "outage_scenarios": scenarios,
            "m_pm": m_pm,
            "m_station": m_station,
            "coverage": 1.0,
            "n_samples": n,
        }

    def evaluate(
        self,
        test_df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates Pipeline A on test data against ground-truth BAM.

        Reports:
        - Overall metrics (N=len(test_df)).
        - Subgroup metrics for missing PM points (m_pm == False).
        - Subgroup metrics for observed PM points (m_pm == True).
        - Scenario-specific metrics (Scenarios 1, 2, 3, 4 where present).
        """
        if self.target_col not in test_df.columns:
            raise ValueError(f"Target column '{self.target_col}' missing from test_df.")

        diag = self.predict_with_diagnostics(test_df, mask=mask)
        preds = diag["predictions"]
        y_true = test_df[self.target_col].to_numpy(dtype=np.float32)

        valid_mask = ~np.isnan(y_true)
        overall_metrics = compute_regression_metrics(y_true[valid_mask], preds[valid_mask])

        # Evaluate on missing PM subset
        m_pm = diag["m_pm"]
        missing_mask = valid_mask & (~m_pm)
        if np.any(missing_mask):
            missing_metrics = compute_regression_metrics(y_true[missing_mask], preds[missing_mask])
        else:
            missing_metrics = None

        # Evaluate on observed PM subset
        observed_mask = valid_mask & m_pm
        if np.any(observed_mask):
            observed_metrics = compute_regression_metrics(y_true[observed_mask], preds[observed_mask])
        else:
            observed_metrics = None

        # Scenario metrics
        scenarios = diag["outage_scenarios"]
        scenario_metrics = {}
        for s_id in [1, 2, 3, 4]:
            s_mask = valid_mask & (scenarios == s_id)
            if np.any(s_mask):
                scenario_metrics[f"scenario_{s_id}"] = {
                    "count": int(np.sum(s_mask)),
                    "metrics": compute_regression_metrics(y_true[s_mask], preds[s_mask]),
                }

        return {
            "pipeline": "Pipeline A (Impute -> Calibrate)",
            "overall_metrics": overall_metrics,
            "missing_pm_metrics": missing_metrics,
            "observed_pm_metrics": observed_metrics,
            "scenario_metrics": scenario_metrics,
            "diagnostics": diag,
        }


# =====================================================================
# Pipeline B: Sequential Calibrate -> Impute
# =====================================================================

class SequentialCalibrateThenImpute:
    """
    Pipeline B: Sequential Calibrate -> Impute.

    Workflow:
    1. Calibrate raw sensor features at observed timestamps:
       Y_hat_cal,t = g_cal(X_t) for m_t^PM = 1
    2. Missing predictions where m_t^PM = 0 are marked as NaN (never evaluated on corrupted sensor data).
    3. Time-series / covariate imputer (fitted strictly on training calibrated predictions)
       reconstructs missing calibrated values:
       Y_hat_B,t = h_recon(Y_hat_cal,t [observed], Covariates_t)
    4. Evaluated against ground-truth BAM.

    Methodological Guarantees:
    - Calibration model g_cal fitted exclusively on train_df.
    - Reconstruction model h_recon fitted exclusively on train_df calibrated predictions + covariates.
    - Ground-truth BAM is NEVER fed to h_recon as an input.
    - Test BAM is never used during prediction or reconstruction.
    - Guarantees 100% prediction availability across all test timestamps.
    """

    def __init__(
        self,
        calibration_model: Optional[Any] = None,
        model_name: str = "Random Forest",
        imputer_type: str = "rf",
        imputer_params: Optional[Dict[str, Any]] = None,
        pm_cols: Optional[List[str]] = None,
        meteo_cols: Optional[List[str]] = None,
        aux_cols: Optional[List[str]] = None,
        calib_cols: Optional[List[str]] = None,
        target_col: str = "bam_reference",
        scale_features: bool = False,
        random_state: int = 42,
    ):
        self.calibration_model = calibration_model
        self.model_name = model_name
        self.imputer_type = imputer_type.lower()
        self.imputer_params = imputer_params or {}
        self.pm_cols = list(pm_cols) if pm_cols is not None else list(DEFAULT_PM_COLS)
        self.meteo_cols = list(meteo_cols) if meteo_cols is not None else list(DEFAULT_METEO_COLS)
        self.aux_cols = list(aux_cols) if aux_cols is not None else list(DEFAULT_AUX_COLS)
        self.calib_feature_cols_ = list(calib_cols) if calib_cols is not None else (self.pm_cols + self.meteo_cols)
        self.target_col = target_col
        self.scale_features = scale_features
        self.random_state = random_state

        # Combined features for tracking
        self.feature_cols_ = self.pm_cols + self.meteo_cols + self.aux_cols
        # Covariates used for reconstruction: meteorology + cyclical harmonics
        self.covariate_cols_ = self.meteo_cols + self.aux_cols

        self.calibration_model_ = None
        self.scaler_ = None
        self.imputer_ = None
        self.train_cal_median_ = None
        self.train_cov_medians_ = None
        self.is_fitted_ = False

    def _build_imputer(self) -> Any:
        """Instantiates the covariate-conditioned reconstruction imputer."""
        if self.imputer_type == "rf":
            params = {"n_estimators": 50, "max_depth": 10, "random_state": self.random_state, "n_jobs": -1}
            params.update(self.imputer_params)
            return RandomForestRegressor(**params)
        elif self.imputer_type == "linear":
            return LinearRegression()
        elif self.imputer_type == "knn":
            params = {"n_neighbors": 5, "weights": "distance"}
            params.update(self.imputer_params)
            return KNNImputer(**params)
        else:
            raise ValueError(f"Unsupported imputer_type for Pipeline B: {self.imputer_type}")

    def _apply_mask_to_df(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Applies missingness mask to dataframe features."""
        df_masked = df.copy()
        n = len(df_masked)

        if mask is None:
            return df_masked

        if isinstance(mask, pd.DataFrame):
            for col in self.feature_cols_:
                if col in mask.columns:
                    col_mask = mask[col].to_numpy(dtype=bool)
                    df_masked.loc[~col_mask, col] = np.nan
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                if mask.shape == (n, len(self.feature_cols_)):
                    mask_bool = mask.astype(bool)
                    for col_idx, col in enumerate(self.feature_cols_):
                        df_masked.loc[~mask_bool[:, col_idx], col] = np.nan
                elif mask.shape == (n, len(self.pm_cols)):
                    mask_bool = mask.astype(bool)
                    for col_idx, col in enumerate(self.pm_cols):
                        df_masked.loc[~mask_bool[:, col_idx], col] = np.nan
            elif mask.ndim == 1:
                mask_bool = mask.astype(bool)
                for col in self.pm_cols:
                    df_masked.loc[~mask_bool, col] = np.nan

        return df_masked

    def fit(
        self,
        train_df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "SequentialCalibrateThenImpute":
        """
        Fits calibration model and reconstruction imputer strictly on train_df.

        Guarantees:
        - g_cal is fitted on train_df complete sensor observations against BAM.
        - h_recon is fitted on train_df covariates to reconstruct calibrated predictions.
        - BAM targets are NEVER fed to h_recon as input features.
        """
        if self.target_col not in train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in train_df.")

        # Ensure required features exist
        for col in self.calib_feature_cols_:
            if col not in train_df.columns:
                raise ValueError(f"Required calibration column missing from train_df: {col}")

        # 1. Fit calibration model on complete training observations
        X_cal_train = train_df[self.calib_feature_cols_].to_numpy(dtype=np.float32)
        y_train = train_df[self.target_col].to_numpy(dtype=np.float32)

        valid_mask = (~np.isnan(X_cal_train).any(axis=1)) & (~np.isnan(y_train))
        X_cal_clean = X_cal_train[valid_mask]
        y_train_clean = y_train[valid_mask]

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_cal_clean = self.scaler_.fit_transform(X_cal_clean)

        if self.calibration_model is not None:
            self.calibration_model_ = self.calibration_model
        else:
            self.calibration_model_ = build_model(self.model_name, random_state=self.random_state)

        self.calibration_model_.fit(X_cal_clean, y_train_clean)

        # 2. Generate training calibrated values (g_cal predictions)
        y_cal_train = self.calibration_model_.predict(X_cal_clean)
        if y_cal_train.ndim > 1:
            y_cal_train = y_cal_train.ravel()

        self.train_cal_median_ = float(np.nanmedian(y_cal_train))

        # 3. Fit reconstruction imputer on covariates to predict calibrated PM
        # Notice: y_train (BAM) is NOT passed as input!
        avail_cov_cols = [c for c in self.covariate_cols_ if c in train_df.columns]
        self.covariate_cols_ = avail_cov_cols
        X_cov_train = train_df.iloc[valid_mask][self.covariate_cols_].to_numpy(dtype=np.float32)

        self.train_cov_medians_ = train_df[self.covariate_cols_].median().to_dict()

        self.imputer_ = self._build_imputer()
        if self.imputer_type in ("rf", "linear"):
            # Supervised fit: Covariates -> Calibrated PM
            self.imputer_.fit(X_cov_train, y_cal_train)
        elif self.imputer_type == "knn":
            # Joint matrix fit: [Calibrated PM, Covariates]
            train_joint_matrix = np.column_stack([y_cal_train, X_cov_train])
            self.imputer_.fit(train_joint_matrix)

        self.is_fitted_ = True
        return self

    def predict_with_diagnostics(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Executes Pipeline B inference:
        1. Calibrates where all PM channels are observed (m_pm == True).
        2. Where PM channels are missing, sets calibrated value to NaN.
        3. Uses reconstruction model to impute missing calibrated predictions.
        4. Guarantees 100% finite prediction coverage without using test BAM.
        """
        if not self.is_fitted_:
            raise RuntimeError("SequentialCalibrateThenImpute must be fitted before predict.")

        n = len(df)
        X_df = df[self.feature_cols_].copy().astype(np.float32) if set(self.feature_cols_).issubset(df.columns) else df.copy()
        X_masked = self._apply_mask_to_df(X_df, mask)

        # Check channel availability
        pm_obs = (~X_masked[self.pm_cols].isna()).to_numpy()
        meteo_obs = (~X_masked[self.meteo_cols].isna()).to_numpy()

        pm_obs_count = pm_obs.sum(axis=1)
        meteo_obs_count = meteo_obs.sum(axis=1)

        n_pm = len(self.pm_cols)
        n_meteo = len(self.meteo_cols)

        m_pm = (pm_obs_count == n_pm)
        m_station = (pm_obs_count == n_pm) & (meteo_obs_count == n_meteo)

        scenarios = np.zeros(n, dtype=int)
        for i in range(n):
            if pm_obs_count[i] == n_pm:
                scenarios[i] = 1
            elif pm_obs_count[i] > 0:
                scenarios[i] = 2
            elif meteo_obs_count[i] > 0:
                scenarios[i] = 3
            else:
                scenarios[i] = 4

        # Step 1: Calibrate observed rows
        calibrated_preds = np.full(n, np.nan, dtype=np.float32)
        if np.any(m_pm):
            X_cal_obs = X_masked.loc[m_pm, self.calib_feature_cols_].to_numpy(dtype=np.float32)
            if self.scale_features and self.scaler_ is not None:
                X_cal_obs = self.scaler_.transform(X_cal_obs)
            preds_obs = self.calibration_model_.predict(X_cal_obs)
            if preds_obs.ndim > 1:
                preds_obs = preds_obs.ravel()
            calibrated_preds[m_pm] = preds_obs

        # Step 2: Impute missing calibrated predictions using reconstruction model
        final_preds = calibrated_preds.copy()
        missing_indices = np.where(~m_pm)[0]

        if len(missing_indices) > 0:
            # Prepare covariates for missing rows
            X_cov_missing = X_masked.iloc[missing_indices][self.covariate_cols_].copy()

            # Handle complete station outages in covariates (Scenario 4 fallback)
            for col in self.covariate_cols_:
                nan_mask = X_cov_missing[col].isna()
                if nan_mask.any():
                    X_cov_missing.loc[nan_mask, col] = self.train_cov_medians_[col]

            X_cov_arr = X_cov_missing.to_numpy(dtype=np.float32)

            if self.imputer_type in ("rf", "linear"):
                imputed_vals = self.imputer_.predict(X_cov_arr)
            elif self.imputer_type == "knn":
                # For KNN: query with NaN target + observed covariates
                test_query = np.column_stack([np.full(len(missing_indices), np.nan), X_cov_arr])
                imputed_vals = self.imputer_.transform(test_query)[:, 0]

            if np.isnan(imputed_vals).any():
                imputed_vals = np.nan_to_num(imputed_vals, nan=self.train_cal_median_)

            final_preds[missing_indices] = imputed_vals

        # Guarantee 100% coverage
        assert not np.isnan(final_preds).any(), "Pipeline B predictions contain NaNs."
        assert len(final_preds) == n

        return {
            "predictions": final_preds,
            "raw_calibrated": calibrated_preds,
            "outage_scenarios": scenarios,
            "m_pm": m_pm,
            "m_station": m_station,
            "coverage": 1.0,
            "n_samples": n,
        }

    def predict(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> np.ndarray:
        return self.predict_with_diagnostics(df, mask=mask)["predictions"]

    def evaluate(
        self,
        test_df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Evaluates Pipeline B on test data against BAM reference."""
        if self.target_col not in test_df.columns:
            raise ValueError(f"Target column '{self.target_col}' missing from test_df.")

        diag = self.predict_with_diagnostics(test_df, mask=mask)
        preds = diag["predictions"]
        y_true = test_df[self.target_col].to_numpy(dtype=np.float32)

        valid_mask = ~np.isnan(y_true)
        overall_metrics = compute_regression_metrics(y_true[valid_mask], preds[valid_mask])

        m_pm = diag["m_pm"]
        missing_mask = valid_mask & (~m_pm)
        missing_metrics = compute_regression_metrics(y_true[missing_mask], preds[missing_mask]) if np.any(missing_mask) else None

        observed_mask = valid_mask & m_pm
        observed_metrics = compute_regression_metrics(y_true[observed_mask], preds[observed_mask]) if np.any(observed_mask) else None

        return {
            "pipeline": "Pipeline B (Calibrate -> Impute)",
            "overall_metrics": overall_metrics,
            "missing_pm_metrics": missing_metrics,
            "observed_pm_metrics": observed_metrics,
            "diagnostics": diag,
        }


# =====================================================================
# Pipeline D: Complete-Case Calibration Reference
# =====================================================================

class CompleteCaseCalibrationReference:
    """
    Pipeline D: Complete-Case Calibration Reference.

    Principles:
    1. Fits calibration model exclusively on complete cases of train_df (no missing sensor channels).
    2. Does NOT perform silent imputation. Missing sensor values produce NO prediction (returns NaN).
    3. Evaluated STRICTLY on its eligible observed subset (m_t^PM = 1).
    4. Prediction availability: (1 - missing_rate) * 100% (e.g. 80% on 20% MCAR; 0% on missing points).
    5. Selection Bias Disclosure: Complete cases may have a different environmental distribution
       than outage periods. It is formally a 'Complete-Case Calibration Reference', not an upper bound.
    """

    def __init__(
        self,
        calibration_model: Optional[Any] = None,
        model_name: str = "Random Forest",
        calib_cols: Optional[List[str]] = None,
        target_col: str = "bam_reference",
        scale_features: bool = False,
        random_state: int = 42,
    ):
        self.calibration_model = calibration_model
        self.model_name = model_name
        self.calib_feature_cols_ = list(calib_cols) if calib_cols is not None else list(DEFAULT_CALIB_COLS)
        self.target_col = target_col
        self.scale_features = scale_features
        self.random_state = random_state

        self.calibration_model_ = None
        self.scaler_ = None
        self.n_train_complete_ = 0
        self.n_train_dropped_ = 0
        self.is_fitted_ = False

    def fit(self, train_df: pd.DataFrame) -> "CompleteCaseCalibrationReference":
        """Fits calibration model strictly on complete training rows."""
        if self.target_col not in train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in train_df.")

        # Complete-case selection
        complete_mask = ~train_df[self.calib_feature_cols_ + [self.target_col]].isna().any(axis=1)
        self.n_train_complete_ = int(np.sum(complete_mask))
        self.n_train_dropped_ = int(len(train_df) - self.n_train_complete_)

        train_complete = train_df.loc[complete_mask]
        X_train = train_complete[self.calib_feature_cols_].to_numpy(dtype=np.float32)
        y_train = train_complete[self.target_col].to_numpy(dtype=np.float32)

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X_train)

        if self.calibration_model is not None:
            self.calibration_model_ = self.calibration_model
        else:
            self.calibration_model_ = build_model(self.model_name, random_state=self.random_state)

        self.calibration_model_.fit(X_train, y_train)
        self.is_fitted_ = True
        return self

    def predict_with_diagnostics(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Generates predictions ONLY for complete observations.
        Missing points receive NaN (no silent imputation).
        """
        if not self.is_fitted_:
            raise RuntimeError("CompleteCaseCalibrationReference must be fitted before predict.")

        n = len(df)
        X_df = df[self.calib_feature_cols_].copy().astype(np.float32)

        # Apply mask
        if mask is not None:
            if isinstance(mask, pd.DataFrame):
                for col in self.calib_feature_cols_:
                    if col in mask.columns:
                        X_df.loc[~mask[col].to_numpy(dtype=bool), col] = np.nan
            elif isinstance(mask, np.ndarray):
                if mask.ndim == 1:
                    # 1D mask: applies to PM channels
                    pm_cols = [c for c in DEFAULT_PM_COLS if c in self.calib_feature_cols_]
                    for col in pm_cols:
                        X_df.loc[~mask.astype(bool), col] = np.nan
                elif mask.ndim == 2:
                    if mask.shape == (n, len(self.calib_feature_cols_)):
                        for idx, col in enumerate(self.calib_feature_cols_):
                            X_df.loc[~mask[:, idx].astype(bool), col] = np.nan

        # Complete-case mask
        eligible_mask = ~X_df.isna().any(axis=1).to_numpy()
        preds = np.full(n, np.nan, dtype=np.float32)

        if np.any(eligible_mask):
            X_eligible = X_df.loc[eligible_mask, self.calib_feature_cols_].to_numpy(dtype=np.float32)
            if self.scale_features and self.scaler_ is not None:
                X_eligible = self.scaler_.transform(X_eligible)
            p = self.calibration_model_.predict(X_eligible)
            if p.ndim > 1:
                p = p.ravel()
            preds[eligible_mask] = p

        coverage = float(np.sum(eligible_mask) / n)

        return {
            "predictions": preds,
            "m_pm": eligible_mask,
            "coverage": coverage,
            "n_eligible": int(np.sum(eligible_mask)),
            "n_dropped": int(n - np.sum(eligible_mask)),
            "selection_bias_warning": (
                "Complete-case observations may exhibit distribution shift relative to "
                "missing outage periods. Evaluated strictly on observed subset."
            ),
        }

    def predict(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> np.ndarray:
        return self.predict_with_diagnostics(df, mask=mask)["predictions"]

    def evaluate(
        self,
        test_df: pd.DataFrame,
        mask: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates Pipeline D strictly on complete cases.
        Reports availability = 0.0% on missing periods and discloses ranking restrictions.
        """
        if self.target_col not in test_df.columns:
            raise ValueError(f"Target column '{self.target_col}' missing from test_df.")

        diag = self.predict_with_diagnostics(test_df, mask=mask)
        preds = diag["predictions"]
        y_true = test_df[self.target_col].to_numpy(dtype=np.float32)

        eligible_mask = diag["m_pm"] & (~np.isnan(y_true))
        if np.any(eligible_mask):
            observed_metrics = compute_regression_metrics(y_true[eligible_mask], preds[eligible_mask])
        else:
            observed_metrics = None

        return {
            "pipeline": "Pipeline D (Complete-Case Calibration Reference)",
            "observed_pm_metrics": observed_metrics,
            "missing_pm_metrics": None,
            "missing_pm_availability": 0.0,
            "ranking_eligibility": (
                "Restricted to matched observed subset (m_pm=1). "
                "Cannot be ranked against full-population pipelines on missing timestamps."
            ),
            "diagnostics": diag,
        }


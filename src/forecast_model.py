"""
Forecasting Model Module
========================
Implements Random Forest based solar generation and household demand 
forecasting. Chosen for interpretability, ease of retraining, and 
competitive accuracy (Kalra et al., 2024).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Tuple, Dict, Optional
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolarForecastModel:
    """Random Forest forecaster for solar generation and household demand."""

    def __init__(self, target: str = "generation", n_estimators: int = 200,
                 max_depth: int = 20, random_state: int = 42):
        """
        Args:
            target: "generation" or "demand"
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.target = target
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.metrics = {}
        self.is_trained = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer temporal and weather features for forecasting.

        Features:
        - Temporal: hour, day_of_week, month, is_weekend
        - Lag: previous 1, 2, 3 period values
        - Rolling: 7-period mean and std
        - Weather: irradiance, temperature, humidity (if available)
        """
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Lag features (requires sorted data)
        df = df.sort_values(["household_id", "timestamp"])
        for lag in [1, 2, 3]:
            df[f"lag_{lag}"] = df.groupby("household_id")["value"].shift(lag)

        # Rolling statistics
        df["rolling_mean_7"] = df.groupby("household_id")["value"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df["rolling_std_7"] = df.groupby("household_id")["value"].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )

        # Weather features (if available)
        for weather_col in ["irradiance", "temperature", "humidity"]:
            if weather_col in df.columns:
                df[f"weather_{weather_col}"] = df[weather_col]

        # Drop rows with NaN from lag features
        df = df.dropna()

        return df

    def prepare_data(self, df: pd.DataFrame, 
                     feature_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and target vector y.

        Args:
            df: DataFrame with engineered features
            feature_cols: List of feature column names (auto-detected if None)

        Returns:
            X, y ready for sklearn
        """
        if feature_cols is None:
            # Auto-detect feature columns (exclude target and metadata)
            exclude = ["timestamp", "household_id", "value", "is_synthetic"]
            feature_cols = [c for c in df.columns if c not in exclude]

        self.feature_names = feature_cols
        X = df[feature_cols]
        y = df["value"]

        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.target} forecast model on {len(X_train)} samples")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Training set metrics (for sanity check — not for evaluation)
        train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        logger.info(f"Training R²: {train_r2:.4f}")

        return {"train_r2": train_r2, "n_samples": len(X_train)}

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model on held-out test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.model.predict(X_test)

        self.metrics = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
            "n_test": len(X_test)
        }

        logger.info(f"Test R²: {self.metrics['r2']:.4f} | MAE: {self.metrics['mae']:.4f}")

        # Check against target
        if self.metrics["r2"] < 0.80:
            logger.warning(f"R² {self.metrics['r2']:.4f} below target 0.80 — consider feature engineering")

        return self.metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.predict(X)

    def save(self, path: str):
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "target": self.target
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load trained model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.target = data["target"]
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class ForecastingPipeline:
    """End-to-end pipeline for training and evaluating both forecasts."""

    def __init__(self, data_loader, random_state: int = 42):
        self.data_loader = data_loader
        self.random_state = random_state
        self.generation_model = None
        self.demand_model = None

    def run(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Run full forecasting pipeline.

        Args:
            df: Merged dataset with timestamp, household_id, generation_kwh, consumption_kwh
            test_size: Fraction of data for testing

        Returns:
            Dictionary with both models' metrics
        """
        results = {}

        # Train generation forecast
        logger.info("=== Training Generation Forecast ===")
        gen_df = df.copy()
        gen_df["value"] = gen_df["generation_kwh"]
        gen_df = SolarForecastModel(target="generation").engineer_features(gen_df)

        gen_model = SolarForecastModel(target="generation", random_state=self.random_state)
        X_gen, y_gen = gen_model.prepare_data(gen_df)
        Xg_train, Xg_test, yg_train, yg_test = train_test_split(
            X_gen, y_gen, test_size=test_size, shuffle=False  # Time series — no shuffle
        )
        gen_model.train(Xg_train, yg_train)
        results["generation"] = gen_model.evaluate(Xg_test, yg_test)
        self.generation_model = gen_model

        # Train demand forecast
        logger.info("=== Training Demand Forecast ===")
        dem_df = df.copy()
        dem_df["value"] = dem_df["consumption_kwh"]
        dem_df = SolarForecastModel(target="demand").engineer_features(dem_df)

        dem_model = SolarForecastModel(target="demand", random_state=self.random_state)
        X_dem, y_dem = dem_model.prepare_data(dem_df)
        Xd_train, Xd_test, yd_train, yd_test = train_test_split(
            X_dem, y_dem, test_size=test_size, shuffle=False
        )
        dem_model.train(Xd_train, yd_train)
        results["demand"] = dem_model.evaluate(Xd_test, yd_test)
        self.demand_model = dem_model

        return results


if __name__ == "__main__":
    print("Forecasting model module ready.")
    print("Usage: from src.forecast_model import SolarForecastModel, ForecastingPipeline")

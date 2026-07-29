"""
Data Loader Module
==================
Handles downloading, cleaning, and preprocessing of public Australian solar
and weather datasets for the benchmarking prototype.

Governance: All data sources are public, de-identified or synthetic.
No live household data is used.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolarDataLoader:
    """Load and preprocess Australian solar generation and household demand data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_ausgrid_data(self, sample: bool = False) -> pd.DataFrame:
        """
        Load Ausgrid Solar Home Electricity Dataset.

        Args:
            sample: If True, load small sample for testing.

        Returns:
            DataFrame with columns: timestamp, household_id, generation_kwh, 
            consumption_kwh, grid_import_kwh, grid_export_kwh
        """
        if sample:
            path = self.data_dir / "sample" / "sample_ausgrid_10homes.csv"
            logger.info("Loading sample Ausgrid data (10 homes)")
        else:
            path = self.raw_dir / "ausgrid_solar_home" / "solar_home_data.csv"
            logger.info("Loading full Ausgrid dataset")

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. "
                "Download from: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data"
            )

        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = self._validate_ausgrid(df)
        return df

    def load_aemo_generation(self) -> pd.DataFrame:
        """Load AEMO rooftop PV generation data."""
        path = self.raw_dir / "aemo_rooftop_pv" / "rooftop_pv.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"AEMO data not found at {path}. "
                "Download from: https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/system-forecasting"
            )
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df

    def load_weather_data(self, station_id: str = "066062") -> pd.DataFrame:
        """
        Load BOM weather data for a given station.

        Args:
            station_id: BOM station identifier (default: Sydney Observatory)
        """
        path = self.raw_dir / "bom_weather" / f"weather_{station_id}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Weather data not found at {path}. "
                "Download from: http://www.bom.gov.au/climate/data/"
            )
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df

    def generate_synthetic_profiles(self, n_profiles: int = 50, 
                                    random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic household profiles for gap-filling.

        Uses statistical matching to real household distributions.
        All outputs are clearly flagged as synthetic.

        Args:
            n_profiles: Number of synthetic profiles to generate
            random_state: Random seed for reproducibility

        Returns:
            DataFrame with synthetic flag and household profiles
        """
        np.random.seed(random_state)
        logger.info(f"Generating {n_profiles} synthetic household profiles")

        # Base distributions derived from Ausgrid summary statistics
        base_consumption = np.random.normal(20, 8, n_profiles)  # kWh/day
        base_generation = np.random.normal(15, 10, n_profiles)   # kWh/day

        profiles = []
        for i in range(n_profiles):
            profile = {
                "household_id": f"SYNTH_{i:03d}",
                "is_synthetic": True,
                "avg_daily_consumption_kwh": max(0, base_consumption[i]),
                "avg_daily_generation_kwh": max(0, base_generation[i]),
                "battery_capacity_kwh": np.random.choice([0, 5, 10, 13.5]),
                "tariff_type": np.random.choice(["flat", "time_of_use", "solar_feed_in"]),
            }
            profiles.append(profile)

        df = pd.DataFrame(profiles)

        # Save with clear synthetic flag
        output_path = self.data_dir / "synthetic" / "synthetic_profiles_50.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic profiles saved to {output_path}")

        return df

    def merge_datasets(self, demand_df: pd.DataFrame, 
                       generation_df: pd.DataFrame,
                       weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge demand, generation and weather data on timestamp.

        Args:
            demand_df: Household demand data
            generation_df: Solar generation data
            weather_df: Weather observations

        Returns:
            Merged DataFrame ready for feature engineering
        """
        logger.info("Merging datasets on timestamp")

        # Ensure timestamp is datetime
        for df in [demand_df, generation_df, weather_df]:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        merged = demand_df.merge(
            generation_df, on="timestamp", how="inner", suffixes=("", "_gen")
        ).merge(
            weather_df, on="timestamp", how="left", suffixes=("", "_weather")
        )

        logger.info(f"Merged dataset shape: {merged.shape}")
        return merged

    def _validate_ausgrid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean Ausgrid data."""
        required_cols = ["timestamp", "household_id", "generation_kwh", "consumption_kwh"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Remove negative values (data errors)
        for col in ["generation_kwh", "consumption_kwh"]:
            df[col] = df[col].clip(lower=0)

        # Remove extreme outliers (> 99.5th percentile)
        for col in ["generation_kwh", "consumption_kwh"]:
            threshold = df[col].quantile(0.995)
            df = df[df[col] <= threshold]

        logger.info(f"Validated Ausgrid data: {len(df)} records, {df['household_id'].nunique()} households")
        return df

    def save_processed(self, df: pd.DataFrame, filename: str):
        """Save processed data to processed directory."""
        path = self.processed_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Processed data saved to {path}")


if __name__ == "__main__":
    # Example usage
    loader = SolarDataLoader()

    # Generate synthetic profiles (no real data needed for skeleton)
    synthetic = loader.generate_synthetic_profiles(n_profiles=10)
    print(synthetic.head())

    print("\nData loader module ready. Download real datasets to raw/ directory for full pipeline.")

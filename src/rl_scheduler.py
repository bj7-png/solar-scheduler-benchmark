"""
Learned Scheduler Module (Reinforcement Learning)
===============================================
Implements a reinforcement learning based battery scheduler using
the Gymnasium interface and stable-baselines3 (PPO).

If RL fails to converge within the sprint time budget, falls back to
a simpler supervised ML policy (gradient boosting).
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatteryEnv(gym.Env):
    """
    Custom Gymnasium environment for battery scheduling.

    State space: [soc, forecast_generation, forecast_demand, grid_price, hour, day_of_week]
    Action space: Discrete(3) — [discharge, maintain, charge]
    Reward: Negative cost (minimise grid import + maximise self-consumption)
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 battery_capacity: float = 10.0,
                 max_charge_rate: float = 5.0,
                 max_discharge_rate: float = 5.0,
                 efficiency: float = 0.95,
                 min_soc: float = 1.0):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(data)

        # Battery parameters
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.efficiency = efficiency
        self.min_soc = min_soc
        self.soc = battery_capacity * 0.5

        # Action space: 0=discharge, 1=maintain, 2=charge
        self.action_space = spaces.Discrete(3)

        # State space: 6 continuous features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([battery_capacity, 20, 20, 1.0, 23, 6]),
            dtype=np.float32
        )

        # Decision log for traceability
        self.decision_log = []

    def _get_obs(self) -> np.ndarray:
        """Get current state observation."""
        row = self.data.iloc[self.current_step]
        return np.array([
            self.soc,
            row["generation_forecast"],
            row["demand_forecast"],
            row["grid_price"],
            row["hour"],
            row["day_of_week"]
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = self.battery_capacity * 0.5
        self.decision_log = []
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.

        Args:
            action: 0=discharge, 1=maintain, 2=charge

        Returns:
            observation, reward, terminated, truncated, info
        """
        row = self.data.iloc[self.current_step]
        gen = row["generation_forecast"]
        dem = row["demand_forecast"]
        price = row["grid_price"]

        power = 0.0

        if action == 2:  # Charge
            excess = gen - dem
            if excess > 0:
                max_charge = min(
                    excess, self.max_charge_rate,
                    (self.battery_capacity - self.soc) / self.efficiency
                )
                if max_charge > 0:
                    self.soc += max_charge * self.efficiency
                    power = max_charge

        elif action == 0:  # Discharge
            if dem > gen and self.soc > self.min_soc:
                shortfall = dem - gen
                max_discharge = min(
                    shortfall, self.max_discharge_rate,
                    (self.soc - self.min_soc) * self.efficiency
                )
                if max_discharge > 0:
                    self.soc -= max_discharge / self.efficiency
                    power = -max_discharge

        # Calculate reward: negative cost
        net_demand = dem - gen + power  # Positive = need grid, Negative = export
        if net_demand > 0:
            cost = net_demand * price
        else:
            cost = net_demand * self.data.iloc[self.current_step].get("feed_in_tariff", 0.08)

        reward = -cost  # Minimise cost = maximise reward

        # Log decision
        self.decision_log.append({
            "step": self.current_step,
            "action": ["discharge", "maintain", "charge"][action],
            "power_kw": power,
            "soc": self.soc,
            "net_demand": net_demand,
            "cost": cost,
            "reward": reward
        })

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def get_decision_log(self) -> pd.DataFrame:
        """Return full decision log for audit."""
        return pd.DataFrame(self.decision_log)


class RLScheduler:
    """Wrapper for RL-based scheduling with fallback to supervised ML."""

    def __init__(self, 
                 model_type: str = "ppo",
                 total_timesteps: int = 100000,
                 fallback_model = None):
        """
        Args:
            model_type: "ppo" or "supervised" (fallback)
            total_timesteps: Training budget for RL
            fallback_model: Pre-trained supervised model for fallback
        """
        self.model_type = model_type
        self.total_timesteps = total_timesteps
        self.model = None
        self.env = None
        self.fallback_model = fallback_model
        self.is_trained = False
        self.training_log = []

    def train(self, train_data: pd.DataFrame, 
              validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the learned scheduler.

        Args:
            train_data: Training dataset with forecasts and prices
            validation_data: Optional validation set for early stopping

        Returns:
            Training metrics
        """
        if self.model_type == "ppo":
            return self._train_ppo(train_data, validation_data)
        else:
            return self._train_supervised(train_data)

    def _train_ppo(self, train_data: pd.DataFrame, 
                   validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """Train PPO agent."""
        logger.info(f"Training PPO agent for {self.total_timesteps} timesteps")

        self.env = BatteryEnv(train_data)

        # Custom callback to log training progress
        class TrainingLogger(BaseCallback):
            def __init__(self, log_freq=1000):
                super().__init__()
                self.log_freq = log_freq

            def _on_step(self) -> bool:
                if self.n_calls % self.log_freq == 0:
                    logger.info(f"PPO step {self.n_calls}: mean_reward={self.locals['rewards'].mean():.4f}")
                return True

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=TrainingLogger(log_freq=10000)
        )

        self.is_trained = True

        # Evaluate on training data
        mean_reward = self._evaluate_policy(self.env)

        metrics = {
            "model_type": "ppo",
            "total_timesteps": self.total_timesteps,
            "mean_reward": mean_reward,
            "converged": mean_reward > -5.0  # Heuristic threshold
        }

        logger.info(f"PPO training complete. Mean reward: {mean_reward:.4f}")

        if not metrics["converged"]:
            logger.warning("PPO did not converge — will use fallback in benchmark")

        return metrics

    def _train_supervised(self, train_data: pd.DataFrame) -> Dict:
        """Train supervised ML policy as fallback."""
        logger.info("Training supervised fallback model (Gradient Boosting)")

        # Create labels: optimal action from hindsight (simplified)
        # In practice, this would use dynamic programming or expert policy
        X = train_data[["soc", "generation_forecast", "demand_forecast", 
                        "grid_price", "hour", "day_of_week"]].values

        # Simplified label: charge if excess generation, discharge if expensive grid
        y = np.zeros(len(train_data))
        for i, row in train_data.iterrows():
            if row["generation_forecast"] > row["demand_forecast"]:
                y[i] = 2  # charge
            elif row["grid_price"] > 0.30:
                y[i] = 0  # discharge
            else:
                y[i] = 1  # maintain

        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        self.fallback_model = model
        self.is_trained = True

        metrics = {
            "model_type": "supervised_gb",
            "n_estimators": 100,
            "train_score": model.score(X, y)
        }

        logger.info(f"Supervised model trained. R²: {metrics['train_score']:.4f}")
        return metrics

    def _evaluate_policy(self, env: BatteryEnv, n_episodes: int = 1) -> float:
        """Evaluate current policy on environment."""
        total_reward = 0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / n_episodes

    def predict(self, state: np.ndarray) -> int:
        """Predict action for a given state."""
        if self.model_type == "ppo" and self.model is not None:
            action, _ = self.model.predict(state, deterministic=True)
            return int(action)
        elif self.fallback_model is not None:
            action = self.fallback_model.predict(state.reshape(1, -1))[0]
            return int(round(action))
        else:
            raise RuntimeError("No model trained")

    def run_simulation(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Run learned scheduler over test data."""
        logger.info(f"Running {self.model_type} simulation over {len(test_data)} timesteps")

        env = BatteryEnv(test_data)
        obs, _ = env.reset()

        for _ in range(len(test_data)):
            action = self.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            if done:
                break

        return env.get_decision_log()

    def save(self, path: str):
        """Save trained model."""
        if self.model_type == "ppo" and self.model is not None:
            self.model.save(path)
            logger.info(f"PPO model saved to {path}")
        elif self.fallback_model is not None:
            import joblib
            joblib.dump(self.fallback_model, path)
            logger.info(f"Supervised model saved to {path}")

    def load(self, path: str, model_type: str = "ppo"):
        """Load trained model."""
        self.model_type = model_type
        if model_type == "ppo":
            self.model = PPO.load(path)
        else:
            import joblib
            self.fallback_model = joblib.load(path)
        self.is_trained = True


if __name__ == "__main__":
    print("RL scheduler module ready.")
    print("Usage: from src.rl_scheduler import RLScheduler, BatteryEnv")
    print("
Models supported:")
    print("  - PPO (stable-baselines3)")
    print("  - Gradient Boosting (supervised fallback)")

"""
Rule-Based Scheduler Module
===========================
Implements a transparent, threshold-based battery scheduling baseline.
Every rule is documented and inspectable — no black-box logic.

This serves as the honest baseline against which the learned scheduler
is benchmarked.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatteryState:
    """Track battery state at each timestep."""
    soc: float          # State of charge (kWh)
    capacity: float     # Total capacity (kWh)
    max_charge_rate: float   # kW
    max_discharge_rate: float  # kW
    efficiency: float   # Round-trip efficiency (0-1)


class RuleBasedScheduler:
    """
    Transparent rule-based battery scheduler.

    Rules (in priority order):
    1. If forecast generation > forecast demand + margin: charge battery
    2. If grid price > discharge_threshold AND soc > min_soc: discharge to grid/home
    3. If forecast demand > forecast generation AND soc > min_soc: discharge to meet demand
    4. Otherwise: maintain current state

    All thresholds are configurable and documented.
    """

    def __init__(self, 
                 battery_capacity: float = 10.0,      # kWh
                 max_charge_rate: float = 5.0,        # kW
                 max_discharge_rate: float = 5.0,     # kW
                 efficiency: float = 0.95,            # Round-trip
                 min_soc: float = 1.0,                # kWh (reserve)
                 charge_margin: float = 0.5,          # kW (generation must exceed demand by this)
                 discharge_threshold_price: float = 0.35,  # $/kWh
                 feed_in_tariff: float = 0.08):       # $/kWh
        """
        Args:
            battery_capacity: Battery capacity in kWh
            max_charge_rate: Maximum charge rate in kW
            max_discharge_rate: Maximum discharge rate in kW
            efficiency: Round-trip efficiency (0-1)
            min_soc: Minimum state of charge to maintain (kWh)
            charge_margin: Generation must exceed demand by this amount to trigger charging
            discharge_threshold_price: Grid price above which discharging is profitable
            feed_in_tariff: Price received for exporting to grid ($/kWh)
        """
        self.battery = BatteryState(
            soc=battery_capacity * 0.5,  # Start at 50% SOC
            capacity=battery_capacity,
            max_charge_rate=max_charge_rate,
            max_discharge_rate=max_discharge_rate,
            efficiency=efficiency
        )
        self.min_soc = min_soc
        self.charge_margin = charge_margin
        self.discharge_threshold_price = discharge_threshold_price
        self.feed_in_tariff = feed_in_tariff

        # Decision log for traceability
        self.decision_log = []

    def schedule(self, 
                 forecast_generation: float,   # kW
                 forecast_demand: float,       # kW
                 grid_price: float,          # $/kWh
                 timestamp: pd.Timestamp) -> Dict:
        """
        Apply transparent scheduling rules for one timestep.

        Args:
            forecast_generation: Predicted solar generation (kW)
            forecast_demand: Predicted household demand (kW)
            grid_price: Current grid electricity price ($/kWh)
            timestamp: Current timestep

        Returns:
            Decision dictionary with action, power, and reasoning
        """
        battery = self.battery
        decision = {
            "timestamp": timestamp,
            "forecast_generation": forecast_generation,
            "forecast_demand": forecast_demand,
            "grid_price": grid_price,
            "soc_before": battery.soc,
            "action": "maintain",
            "power_kw": 0.0,
            "reasoning": "",
            "soc_after": battery.soc
        }

        # Rule 1: Charge if generation exceeds demand by margin
        if forecast_generation > forecast_demand + self.charge_margin:
            excess = forecast_generation - forecast_demand
            max_charge = min(
                excess,
                battery.max_charge_rate,
                (battery.capacity - battery.soc) / battery.efficiency
            )
            if max_charge > 0 and battery.soc < battery.capacity:
                actual_charge = max_charge * battery.efficiency
                battery.soc += actual_charge
                decision["action"] = "charge"
                decision["power_kw"] = max_charge
                decision["reasoning"] = (
                    f"Rule 1: Generation ({forecast_generation:.2f}kW) > "
                    f"Demand ({forecast_demand:.2f}kW) + Margin ({self.charge_margin}kW). "
                    f"Excess solar stored in battery."
                )

        # Rule 2: Discharge if grid price is high and we have reserve
        elif (grid_price > self.discharge_threshold_price and 
              battery.soc > self.min_soc and
              forecast_demand > forecast_generation):
            needed = forecast_demand - forecast_generation
            max_discharge = min(
                needed,
                battery.max_discharge_rate,
                (battery.soc - self.min_soc) * battery.efficiency
            )
            if max_discharge > 0:
                actual_discharge = max_discharge / battery.efficiency
                battery.soc -= actual_discharge
                decision["action"] = "discharge"
                decision["power_kw"] = -max_discharge
                decision["reasoning"] = (
                    f"Rule 2: Grid price (${grid_price:.3f}/kWh) > Threshold "
                    f"(${self.discharge_threshold_price:.3f}/kWh) and SOC "
                    f"({battery.soc:.2f}kWh) > Reserve ({self.min_soc}kWh). "
                    f"Discharge to avoid expensive grid import."
                )

        # Rule 3: Discharge to meet demand if generation insufficient
        elif (forecast_demand > forecast_generation and 
              battery.soc > self.min_soc):
            shortfall = forecast_demand - forecast_generation
            max_discharge = min(
                shortfall,
                battery.max_discharge_rate,
                (battery.soc - self.min_soc) * battery.efficiency
            )
            if max_discharge > 0:
                actual_discharge = max_discharge / battery.efficiency
                battery.soc -= actual_discharge
                decision["action"] = "discharge"
                decision["power_kw"] = -max_discharge
                decision["reasoning"] = (
                    f"Rule 3: Demand ({forecast_demand:.2f}kW) > Generation "
                    f"({forecast_generation:.2f}kW) and SOC sufficient. "
                    f"Discharge to cover shortfall."
                )

        else:
            decision["reasoning"] = (
                f"No rule triggered. Generation={forecast_generation:.2f}kW, "
                f"Demand={forecast_demand:.2f}kWh, Price=${grid_price:.3f}, "
                f"SOC={battery.soc:.2f}kWh."
            )

        decision["soc_after"] = battery.soc
        self.decision_log.append(decision)

        return decision

    def run_simulation(self, 
                       forecasts: pd.DataFrame,
                       prices: pd.Series) -> pd.DataFrame:
        """
        Run rule-based scheduler over a full forecast horizon.

        Args:
            forecasts: DataFrame with columns [timestamp, generation_forecast, demand_forecast]
            prices: Series of grid prices indexed by timestamp

        Returns:
            DataFrame of decisions with reasoning
        """
        logger.info(f"Running rule-based simulation over {len(forecasts)} timesteps")

        results = []
        for _, row in forecasts.iterrows():
            decision = self.schedule(
                forecast_generation=row["generation_forecast"],
                forecast_demand=row["demand_forecast"],
                grid_price=prices.loc[row["timestamp"]],
                timestamp=row["timestamp"]
            )
            results.append(decision)

        df = pd.DataFrame(results)
        logger.info(f"Simulation complete. Actions: {df['action'].value_counts().to_dict()}")
        return df

    def get_decision_log(self) -> pd.DataFrame:
        """Return full decision log for audit and traceability."""
        return pd.DataFrame(self.decision_log)

    def explain_last_decision(self) -> str:
        """Return human-readable explanation of the last decision."""
        if not self.decision_log:
            return "No decisions made yet."
        last = self.decision_log[-1]
        return (
            f"At {last['timestamp']}: {last['action'].upper()} "
            f"({last['power_kw']:.2f}kW). Reason: {last['reasoning']}"
        )

    def reset(self):
        """Reset battery state and decision log."""
        self.battery.soc = self.battery.capacity * 0.5
        self.decision_log = []
        logger.info("Scheduler reset to initial state")


if __name__ == "__main__":
    print("Rule-based scheduler module ready.")
    print("Usage: from src.rule_scheduler import RuleBasedScheduler")
    print("
Transparent rules:")
    print("  1. Charge when generation > demand + margin")
    print("  2. Discharge when grid price > threshold AND demand > generation")
    print("  3. Discharge to cover shortfall when generation < demand")
    print("  4. Maintain otherwise")

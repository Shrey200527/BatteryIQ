"""
Battery Service for BatteryIQ Dashboard Backend.

This service handles battery data retrieval and validation, integrating
with the existing Phase 2 frozen datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add src to path for imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class BatteryService:
    """Service for battery data operations."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.features_file = self.project_root / "data" / "features" / "battery_cycle_features.csv"
        self.targets_file = self.project_root / "data" / "features" / "battery_cycle_targets.csv"
        self.rul_file = self.project_root / "data" / "features" / "rul_targets.csv"
        
        # Load data on initialization
        self._load_data()
    
    def _load_data(self):
        """Load frozen Phase 2 datasets."""
        if not self.features_file.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_file}")
        if not self.targets_file.exists():
            raise FileNotFoundError(f"Targets file not found: {self.targets_file}")
        if not self.rul_file.exists():
            raise FileNotFoundError(f"RUL file not found: {self.rul_file}")
        
        self.features_df = pd.read_csv(self.features_file)
        self.targets_df = pd.read_csv(self.targets_file)
        self.rul_df = pd.read_csv(self.rul_file)
        
        # Merge for convenience
        self.merged_df = self.features_df.merge(
            self.targets_df,
            on=["battery_id", "cycle"],
            how="inner"
        )
    
    def get_available_batteries(self) -> List[str]:
        """Get list of available battery IDs."""
        return sorted(self.features_df["battery_id"].unique().tolist())
    
    def get_battery_info(self, battery_id: str) -> Dict:
        """Get information about a specific battery."""
        if battery_id not in self.get_available_batteries():
            raise ValueError(f"Battery {battery_id} is not available")
        
        battery_data = self.features_df[self.features_df["battery_id"] == battery_id]
        rul_data = self.rul_df[self.rul_df["battery_id"] == battery_id]
        
        total_cycles = len(battery_data)
        is_rul_valid = rul_data["rul_valid"].any()
        
        return {
            "battery_id": battery_id,
            "total_cycles": total_cycles,
            "is_rul_valid": is_rul_valid
        }
    
    def get_battery_detail(self, battery_id: str) -> Dict:
        """Get detailed information about a specific battery."""
        if battery_id not in self.get_available_batteries():
            raise ValueError(f"Battery {battery_id} is not available")
        
        battery_data = self.merged_df[self.merged_df["battery_id"] == battery_id]
        rul_data = self.rul_df[self.rul_df["battery_id"] == battery_id]
        
        total_cycles = len(battery_data)
        is_rul_valid = rul_data["rul_valid"].any()
        soh_min = float(battery_data["soh"].min())
        soh_max = float(battery_data["soh"].max())
        available_cycles = sorted(battery_data["cycle"].tolist())
        
        return {
            "battery_id": battery_id,
            "total_cycles": total_cycles,
            "is_rul_valid": is_rul_valid,
            "soh_range": {"min": soh_min, "max": soh_max},
            "available_cycles": available_cycles
        }
    
    def get_cycle_features(self, battery_id: str, cycle: int) -> Dict:
        """Get features for a specific battery cycle."""
        if battery_id not in self.get_available_batteries():
            raise ValueError(f"Battery {battery_id} is not available")
        
        cycle_data = self.features_df[
            (self.features_df["battery_id"] == battery_id) & 
            (self.features_df["cycle"] == cycle)
        ]
        
        if cycle_data.empty:
            raise ValueError(f"Cycle {cycle} not found for battery {battery_id}")
        
        # Return only the frozen feature columns
        frozen_features = [
            "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
            "current_mean", "current_min", "current_max",
            "temperature_mean", "temperature_max",
            "current_load_mean", "voltage_load_mean",
            "cycle_time", "voltage_change", "temperature_change", "temperature_rolling_mean"
        ]
        
        feature_dict = cycle_data[frozen_features].iloc[0].to_dict()
        # Convert to float for JSON serialization
        feature_dict = {k: float(v) for k, v in feature_dict.items()}
        
        return feature_dict
    
    def validate_battery_id(self, battery_id: str) -> bool:
        """Validate if battery ID exists."""
        return battery_id in self.get_available_batteries()
    
    def validate_cycle(self, battery_id: str, cycle: int) -> bool:
        """Validate if cycle exists for battery."""
        if not self.validate_battery_id(battery_id):
            return False
        
        cycle_data = self.features_df[
            (self.features_df["battery_id"] == battery_id) & 
            (self.features_df["cycle"] == cycle)
        ]
        
        return not cycle_data.empty
    
    def get_consecutive_cycles(self, battery_id: str, start_cycle: int, count: int = 10) -> List[Dict]:
        """Get consecutive cycles for sequence models."""
        if not self.validate_battery_id(battery_id):
            raise ValueError(f"Battery {battery_id} is not available")
        
        # Check if all consecutive cycles exist
        expected_cycles = list(range(start_cycle, start_cycle + count))
        
        cycle_data = self.features_df[
            (self.features_df["battery_id"] == battery_id) & 
            (self.features_df["cycle"].isin(expected_cycles))
        ].sort_values("cycle")
        
        if len(cycle_data) != count:
            missing = set(expected_cycles) - set(cycle_data["cycle"].tolist())
            raise ValueError(f"Missing cycles for sequence: {missing}")
        
        # Verify they are consecutive
        actual_cycles = cycle_data["cycle"].tolist()
        if actual_cycles != expected_cycles:
            raise ValueError(f"Cycles are not consecutive: {actual_cycles}")
        
        # Get frozen features for each cycle
        frozen_features = [
            "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
            "current_mean", "current_min", "current_max",
            "temperature_mean", "temperature_max",
            "current_load_mean", "voltage_load_mean",
            "cycle_time", "voltage_change", "temperature_change", "temperature_rolling_mean"
        ]
        
        cycles_features = []
        for _, row in cycle_data.iterrows():
            feature_dict = row[frozen_features].to_dict()
            feature_dict = {k: float(v) for k, v in feature_dict.items()}
            cycles_features.append(feature_dict)
        
        return cycles_features


# Global service instance
_battery_service = None

def get_battery_service() -> BatteryService:
    """Get or create battery service instance."""
    global _battery_service
    if _battery_service is None:
        _battery_service = BatteryService()
    return _battery_service
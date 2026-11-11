"""
AIRWISE Dashboard: Input Validation and Range Checks
"""

from typing import Dict, List, Tuple
from dashboard.config import PHYSICAL_BOUNDS


def validate_sensor_inputs(
    pa_cf1: float,
    pa_atm: float,
    pa_alt: float,
    rh: float,
    temp: float,
) -> Tuple[bool, List[str]]:
    """
    Validates user-selected sensor values against physical realistic bounds
    and known training set distribution ranges.
    """
    warnings = []
    is_valid = True

    # Check non-negativity
    if pa_cf1 < PHYSICAL_BOUNDS["pm25_min"] or pa_atm < PHYSICAL_BOUNDS["pm25_min"] or pa_alt < PHYSICAL_BOUNDS["pm25_min"]:
        warnings.append("Raw optical PM2.5 readings cannot be negative.")
        is_valid = False

    # Check extreme upper bounds
    if pa_cf1 > PHYSICAL_BOUNDS["pm25_max"] or pa_atm > PHYSICAL_BOUNDS["pm25_max"]:
        warnings.append(
            f"Sensor readings exceed {PHYSICAL_BOUNDS['pm25_max']} µg/m³; optical sensor is likely in complete optical saturation."
        )

    # Relative humidity check
    if rh < PHYSICAL_BOUNDS["rh_min"] or rh > PHYSICAL_BOUNDS["rh_max"]:
        warnings.append(f"Relative humidity should be between {PHYSICAL_BOUNDS['rh_min']}% and {PHYSICAL_BOUNDS['rh_max']}%.")
        is_valid = False
    elif rh > 85.0:
        warnings.append("High relative humidity (>85%) causes significant hygroscopic particle swelling in PurpleAir nephelometers.")

    # Temperature check
    if temp < PHYSICAL_BOUNDS["temp_min"] or temp > PHYSICAL_BOUNDS["temp_max"]:
        warnings.append(f"Ambient temperature should be between {PHYSICAL_BOUNDS['temp_min']}°C and {PHYSICAL_BOUNDS['temp_max']}°C.")
        is_valid = False

    return is_valid, warnings

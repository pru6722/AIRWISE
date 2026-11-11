"""
AIRWISE Dashboard Configuration
Global design tokens, color palettes, CPCB breakpoints, and platform metadata.
"""

from typing import Dict, List, Tuple, Any

# Platform Branding
APP_NAME = "AIRWISE"
APP_TITLE = "AIRWISE — PM2.5 Research Analytics Platform"
APP_SUBTITLE = "Uncertainty-Aware Joint Calibration, Temporal Sequence Reconstruction & CPCB Regulatory Assessment"
ACADEMIC_CITATION = "Campmier et al. (2023), Atmospheric Measurement Techniques, 16, 4357–4374. DOI: 10.5194/amt-16-4357-2023 | Dryad: 10.6078/D1RQ70 (CC0 1.0)"
JOURNAL_CITATION = "Campmier et al. (2023), Seasonally optimized calibrations improve low-cost sensor performance: long-term field evaluation of PurpleAir sensors in urban and rural India, Atmospheric Measurement Techniques, 16, 4357–4374. DOI: 10.5194/amt-16-4357-2023"
DRYAD_DOI = "10.6078/D1RQ70"
DATASET_LICENSE = "Creative Commons Zero v1.0 Universal (CC0 1.0)"

# Dark Academic Theme Palette
THEME = {
    "bg_main": "#0b111e",        # Deep dark navy
    "bg_card": "#131c2e",        # Slightly lighter navy for cards
    "bg_card_hover": "#1a253c",  # Interactive card hover
    "bg_sidebar": "#070c14",     # Darkest navy for sidebar
    "border": "#1e293b",         # Subtle slate border
    "border_active": "#00d2c4",  # Active border highlight (teal)
    "text_primary": "#f8fafd",   # Crisp off-white
    "text_secondary": "#94a3b8", # Muted slate for captions/subtitles
    "text_accent": "#00d2c4",    # Teal accent
    "accent_teal": "#00d2c4",    # Primary accent
    "accent_cyan": "#00a8ff",    # Secondary accent
    "accent_amber": "#ffb86c",   # Warning / intermediate
    "accent_rose": "#ff5555",    # High error / severe alert
    "accent_purple": "#bd93f9",  # Deep learning / TCN
}

# Monitoring Sites
SITES: List[str] = ["Delhi", "Hamirpur", "Bangalore"]

SITE_COORDINATES: Dict[str, Dict[str, float]] = {
    "Delhi": {"lat": 28.5450, "lon": 77.1926, "elevation_m": 225.0},
    "Hamirpur": {"lat": 31.6862, "lon": 76.5213, "elevation_m": 780.0},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "elevation_m": 920.0},
}

# CPCB Regulatory PM2.5 Breakpoints (CUPS/82/2014-15)
CPCB_CATEGORIES: List[str] = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]

CPCB_BREAKPOINTS: Dict[str, Tuple[float, float]] = {
    "Good": (0.0, 30.0),
    "Satisfactory": (30.0, 60.0),
    "Moderate": (60.0, 90.0),
    "Poor": (90.0, 120.0),
    "Very Poor": (120.0, 250.0),
    "Severe": (250.0, 1000.0),
}

CPCB_COLORS: Dict[str, str] = {
    "Good": "#009966",          # Dark Green
    "Satisfactory": "#99CC33",  # Light Green
    "Moderate": "#FFDE33",      # Yellow
    "Poor": "#FF9933",          # Orange
    "Very Poor": "#CC0033",     # Red
    "Severe": "#7E0023",        # Maroon
}

# Feature Subsets
FEATURE_SETS = {
    "cf1_only": ["pa_cf1", "relative_humidity", "temperature"],
    "atm_only": ["pa_atm", "relative_humidity", "temperature"],
    "alt_only": ["pa_alt", "relative_humidity", "temperature"],
    "all_channels": ["pa_cf1", "pa_atm", "pa_alt", "relative_humidity", "temperature"],
}

# Joint TCN 10 Input Features
JOINT_TCN_FEATURES: List[str] = [
    "pa_cf1",
    "pa_atm",
    "pa_alt",
    "relative_humidity",
    "temperature",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "time_diff_hours",
]

# Physical Sensor Boundaries for Interactive Controls
PHYSICAL_BOUNDS = {
    "pm25_min": 0.0,
    "pm25_max": 800.0,
    "rh_min": 5.0,
    "rh_max": 100.0,
    "temp_min": -5.0,
    "temp_max": 50.0,
}

# Reconciled Dataset Partition Counts (Authoritative 60/10/10/20 Chronological Split)
DATASET_SPLITS: Dict[str, int] = {
    "train": 11149,
    "validation": 1857,
    "conformal": 1858,
    "test": 3720,
    "total": 18584,
}

# Phase 3C Four Comparative Pipelines
PIPELINES: Dict[str, Dict[str, Any]] = {
    "A": {
        "code": "A",
        "name": "Pipeline A: Impute → Calibrate",
        "short_name": "Pipeline A (Seq Impute)",
        "color": "#00a8ff",  # Vibrant cyan
        "border_color": "#0284c7",
        "description": "Sequential pipeline: Missing nephelometer channels are first imputed, then calibrated via Random Forest.",
        "paradigm": "Sequential (Impute First)",
        "imputer": "KNN / Iterative Imputer",
        "regressor": "Random Forest",
    },
    "B": {
        "code": "B",
        "name": "Pipeline B: Calibrate → Impute",
        "short_name": "Pipeline B (Seq Calibrate)",
        "color": "#00d2c4",  # Teal accent
        "border_color": "#0f766e",
        "description": "Sequential pipeline: Observed nephelometer channels are calibrated first, missing calibrated estimates are reconstructed.",
        "paradigm": "Sequential (Calibrate First)",
        "imputer": "Reconstruction Imputer",
        "regressor": "Random Forest",
    },
    "C": {
        "code": "C",
        "name": "Pipeline C: Joint Causal TCN v2",
        "short_name": "Pipeline C (Joint TCN v2)",
        "color": "#bd93f9",  # Lavender purple
        "border_color": "#7c3aed",
        "description": "Deep multi-task architecture: Shared causal temporal encoder simultaneously supervising calibration and reconstruction.",
        "paradigm": "Joint Multi-Task Deep Sequence",
        "imputer": "Reconstruction Head (Multi-Task)",
        "regressor": "Calibration Head (Multi-Task)",
    },
    "D": {
        "code": "D",
        "name": "Pipeline D: Complete-Case Reference",
        "short_name": "Pipeline D (Complete Case)",
        "color": "#ffb86c",  # Warm amber
        "border_color": "#d97706",
        "description": "Baseline reference: Standard calibration evaluated strictly on complete-case observations without imputation (N=2,951).",
        "paradigm": "Complete-Case Only (No Imputation)",
        "imputer": "None",
        "regressor": "Random Forest (Complete Cases)",
    },
}

# Phase 3C Nine Missingness Scenarios
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "MCAR_20pct": {
        "id": "MCAR_20pct",
        "name": "20% MCAR",
        "category": "MCAR",
        "badge_color": "#00d2c4",
        "description": "20% Completely At Random Bernoulli channel missingness across PurpleAir nephelometer channels.",
    },
    "Block_1h": {
        "id": "Block_1h",
        "name": "Block Outages (1h)",
        "category": "Block Outage",
        "badge_color": "#00a8ff",
        "description": "Nominal 1-hour sensor outage bursts randomly inserted into test segments.",
    },
    "Block_3h": {
        "id": "Block_3h",
        "name": "Block Outages (3h)",
        "category": "Block Outage",
        "badge_color": "#00a8ff",
        "description": "Nominal 3-hour sensor outage bursts randomly inserted into test segments.",
    },
    "Block_6h": {
        "id": "Block_6h",
        "name": "Block Outages (6h)",
        "category": "Block Outage",
        "badge_color": "#00a8ff",
        "description": "Nominal 6-hour sensor outage bursts randomly inserted into test segments.",
    },
    "Block_12h": {
        "id": "Block_12h",
        "name": "Block Outages (12h)",
        "category": "Block Outage",
        "badge_color": "#00a8ff",
        "description": "Nominal 12-hour sensor outage bursts randomly inserted into test segments.",
    },
    "Block_24h": {
        "id": "Block_24h",
        "name": "Block Outages (24h Nominal)",
        "category": "Block Outage",
        "badge_color": "#ffb86c",
        "description": "Nominal 24-hour diurnal outages (merged continuous duration: 24h to 63h due to overlapping block placement).",
    },
    "MAR_RH": {
        "id": "MAR_RH",
        "name": "MAR-RH (Humidity-Correlated)",
        "category": "MAR",
        "badge_color": "#a78bfa",
        "description": "Missing At Random driven by high ambient relative humidity (RH > 75%) inducing nephelometer failure.",
    },
    "MAR_Lag": {
        "id": "MAR_Lag",
        "name": "MAR-Lag (Persistence-Driven)",
        "category": "MAR",
        "badge_color": "#a78bfa",
        "description": "Missing At Random driven by temporal failure persistence (autoregressive dropout).",
    },
    "MNAR_PM": {
        "id": "MNAR_PM",
        "name": "MNAR-PM (High-Concentration Stress)",
        "category": "MNAR",
        "badge_color": "#ff5555",
        "description": "Oracle stress test: Missingness triggered during unobserved extreme PM2.5 pollution events.",
    },
}

# Evaluation Population Modes
EVALUATION_POPULATIONS: Dict[str, Dict[str, str]] = {
    "overall": {
        "name": "Full Test Population",
        "subtitle": "All N = 3,720 Independent Test Timestamps",
        "description": "Evaluates overall system performance across all test timestamps, blending observed and missing intervals.",
    },
    "missing_pm": {
        "name": "Missing-PM Subgroup",
        "subtitle": "Subgroup where Nephelometer was Missing",
        "description": "Evaluates pure reconstruction fidelity strictly during simulated sensor outages.",
    },
    "common_support": {
        "name": "Common Support (Observed Nephelometer)",
        "subtitle": "Complete Cases Only (m_t = 1, N = 2,951)",
        "description": "Evaluates pure calibration performance on complete-case sensor observations where all channels are present.",
    },
}


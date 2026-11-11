"""
AIRWISE Dashboard: Text and HTML Formatting Utilities
"""

from typing import Optional
from dashboard.config import CPCB_COLORS, THEME


def format_pm25(val: Optional[float], decimals: int = 1) -> str:
    """Formats PM2.5 concentration with standard unit."""
    if val is None or val != val:  # NaN check
        return "N/A"
    return f"{val:.{decimals}f} µg/m³"


def format_metric(val: Optional[float], decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    """Formats numerical metric safely."""
    if val is None or val != val:
        return "N/A"
    return f"{prefix}{val:.{decimals}f}{suffix}"


def get_cpcb_color(category: str) -> str:
    """Returns official hex color for CPCB category."""
    return CPCB_COLORS.get(category, THEME["text_secondary"])


def get_cpcb_badge_html(category: str) -> str:
    """Generates an inline styled HTML pill for CPCB category."""
    color = get_cpcb_color(category)
    return (
        f'<span class="badge-cpcb" style="background-color: {color};">'
        f'{category}</span>'
    )


def get_stability_badge_html(is_stable: bool) -> str:
    """Generates an inline styled HTML badge for uncertainty stability."""
    if is_stable:
        return '<span class="badge-stable">✓ Stable (Single Category)</span>'
    else:
        return '<span class="badge-ambiguous">⚠ Ambiguous (Boundary Spanning)</span>'

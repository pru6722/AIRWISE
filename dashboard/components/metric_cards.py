"""
AIRWISE Dashboard: Metric Card Components
Styled metric cards with dark navy background, teal/cyan typography, and confidence deltas.
"""

from typing import List, Optional, Dict, Any
import streamlit as st
from dashboard.config import THEME


def render_metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    color: Optional[str] = None,
) -> None:
    """Renders a single styled metric card."""
    card_color = color or THEME["accent_teal"]
    sub_html = f'<div class="airwise-card-sub" style="color: {card_color};">{subtitle}</div>' if subtitle else ""
    delta_html = f'<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">{delta}</div>' if delta else ""

    html = f"""
    <div class="airwise-card">
        <div class="airwise-card-title">{title}</div>
        <div class="airwise-card-value" style="color: #f8fafd;">{value}</div>
        {sub_html}
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_row(cards: List[Dict[str, Any]]) -> None:
    """Renders a responsive horizontal row of metric cards."""
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_metric_card(
                title=card.get("title", ""),
                value=card.get("value", ""),
                subtitle=card.get("subtitle"),
                delta=card.get("delta"),
                color=card.get("color"),
            )

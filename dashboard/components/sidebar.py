"""
AIRWISE Dashboard: Sidebar Component
Persistent navigation header, context-aware shared global filters, and research protocol disclosures.
"""

from typing import Tuple, Optional, Dict, Any, List
import datetime
import streamlit as st

from dashboard.config import (
    SITES,
    THEME,
    APP_NAME,
    SCENARIOS,
    PIPELINES,
    EVALUATION_POPULATIONS,
    DATASET_SPLITS,
)

DEFAULT_MIN_DATE = datetime.date(2018, 7, 24)
DEFAULT_MAX_DATE = datetime.date(2021, 1, 11)

DEFAULT_FILTERS: Dict[str, Any] = {
    "site": "All Sites",
    "start_date": DEFAULT_MIN_DATE,
    "end_date": DEFAULT_MAX_DATE,
    "scenario_id": "MCAR_20pct",
    "pipeline_code": "B",
    "population_mode": "overall",
    "metric": "rmse",
}

_WIDGET_KEYS = [
    "_sb_site",
    "_sb_date_range",
    "_sb_scenario",
    "_sb_pipeline",
    "_sb_population",
    "_sb_metric",
]


def init_sidebar_state() -> Dict[str, Any]:
    """Initializes persistent cross-page filter state if absent."""
    if "airwise_filters" not in st.session_state:
        st.session_state["airwise_filters"] = DEFAULT_FILTERS.copy()
    return st.session_state["airwise_filters"]


def get_global_filters() -> Dict[str, Any]:
    """Returns a copy of the currently active cross-page filters."""
    return init_sidebar_state().copy()


def reset_global_filters() -> None:
    """Resets all persistent filters and transient sidebar widget states to defaults."""
    st.session_state["airwise_filters"] = DEFAULT_FILTERS.copy()
    for wk in _WIDGET_KEYS:
        if wk in st.session_state:
            del st.session_state[wk]


def _render_sidebar_controls(
    show_site_filter: bool = True,
    show_date_filter: bool = True,
    show_scenario_filter: bool = False,
    show_pipeline_filter: bool = False,
    show_population_filter: bool = False,
    show_metric_filter: bool = False,
    active_page: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Internal unified sidebar renderer supporting persistent global filters
    while strictly respecting analytical relevance per page.
    """
    filters = init_sidebar_state()

    with st.sidebar:
        # Platform Branding
        active_badge_html = (
            f'<div style="display: inline-block; margin-top: 6px; padding: 2px 8px; '
            f'border-radius: 4px; background-color: rgba(0, 210, 196, 0.14); '
            f'border: 1px solid #00d2c4; color: #00d2c4; font-size: 0.70rem; font-weight: 600;">'
            f'Active Module: {active_page}</div>'
            if active_page
            else ""
        )

        st.markdown(
            f"""
            <div style="padding: 8px 0 14px 0; border-bottom: 1px solid #1e293b; margin-bottom: 14px;">
                <div style="font-size: 1.55rem; font-weight: 800; color: #00d2c4; letter-spacing: -0.5px;">
                    {APP_NAME}
                </div>
                <div style="font-size: 0.76rem; color: #94a3b8; line-height: 1.35; margin-top: 2px;">
                    Adaptive Imputation &amp; Robust Windowed Inference for Sensor Estimation
                </div>
                <div style="font-size: 0.70rem; color: #00a8ff; margin-top: 5px; font-weight: 600;">
                    {DATASET_SPLITS['total']:,} Curated Obs | {len(SITES)} Monitoring Cities
                </div>
                {active_badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        has_any_filter = any([
            show_site_filter,
            show_date_filter,
            show_scenario_filter,
            show_pipeline_filter,
            show_population_filter,
            show_metric_filter,
        ])

        if has_any_filter:
            st.markdown("### Shared Research Filters")

        # 1. Site selector
        if show_site_filter:
            site_options = ["All Sites"] + SITES
            if "_sb_site" not in st.session_state or st.session_state["_sb_site"] not in site_options:
                st.session_state["_sb_site"] = filters["site"] if filters["site"] in site_options else "All Sites"
            selected_site = st.selectbox(
                "Monitoring Station",
                options=site_options,
                key="_sb_site",
                help="Filter visualizations to Bangalore, Delhi, Hamirpur, or pooled test records.",
            )
            filters["site"] = selected_site

        # 2. Temporal range selector
        if show_date_filter:
            if "_sb_date_range" not in st.session_state:
                st.session_state["_sb_date_range"] = (filters["start_date"], filters["end_date"])
            date_range = st.date_input(
                "Temporal Window",
                min_value=DEFAULT_MIN_DATE,
                max_value=DEFAULT_MAX_DATE,
                key="_sb_date_range",
                help="Restrict time-series and exploration views to a chronological date span.",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                s_dt, e_dt = date_range
                if s_dt <= e_dt:
                    filters["start_date"] = s_dt
                    filters["end_date"] = e_dt
                else:
                    filters["start_date"] = e_dt
                    filters["end_date"] = s_dt
            elif isinstance(date_range, tuple) and len(date_range) == 1:
                filters["start_date"] = date_range[0]

        # 3. Missingness scenario selector (Phase 3C 9 scenarios)
        if show_scenario_filter:
            scenario_keys = list(SCENARIOS.keys())
            if "_sb_scenario" not in st.session_state or st.session_state["_sb_scenario"] not in scenario_keys:
                st.session_state["_sb_scenario"] = filters["scenario_id"] if filters["scenario_id"] in scenario_keys else "MCAR_20pct"
            selected_scenario = st.selectbox(
                "Missingness Scenario",
                options=scenario_keys,
                format_func=lambda k: SCENARIOS[k]["name"],
                key="_sb_scenario",
                help="Select one of the 9 benchmark missingness masks (MCAR, Block 1h-24h, MAR, MNAR).",
            )
            filters["scenario_id"] = selected_scenario

        # 4. Pipeline selector (A, B, C, D)
        if show_pipeline_filter:
            pipeline_keys = list(PIPELINES.keys())
            if "_sb_pipeline" not in st.session_state or st.session_state["_sb_pipeline"] not in pipeline_keys:
                st.session_state["_sb_pipeline"] = filters["pipeline_code"] if filters["pipeline_code"] in pipeline_keys else "B"
            selected_pipeline = st.selectbox(
                "Calibration / Reconstruction Pipeline",
                options=pipeline_keys,
                format_func=lambda k: PIPELINES[k]["name"],
                key="_sb_pipeline",
                help="Select primary pipeline for residual, scatter, and time-series diagnostics.",
            )
            filters["pipeline_code"] = selected_pipeline

        # 5. Evaluation population selector (strictly separated)
        if show_population_filter:
            pop_keys = list(EVALUATION_POPULATIONS.keys())
            if "_sb_population" not in st.session_state or st.session_state["_sb_population"] not in pop_keys:
                st.session_state["_sb_population"] = filters["population_mode"] if filters["population_mode"] in pop_keys else "overall"
            selected_pop = st.selectbox(
                "Evaluation Population",
                options=pop_keys,
                format_func=lambda k: EVALUATION_POPULATIONS[k]["name"],
                key="_sb_population",
                help="Populations are strictly isolated: never mix Full Test, Missing-PM Subgroup, and Common Support.",
            )
            filters["population_mode"] = selected_pop
            st.caption(
                "Population Guard: Selected subgroup isolates metrics without blending train/val/test sets."
            )

        # 6. Metric selector
        if show_metric_filter:
            metric_options = ["rmse", "mae", "r2"]
            metric_labels = {
                "rmse": "RMSE (µg/m³)",
                "mae": "MAE (µg/m³)",
                "r2": "R² (Explained Variance)",
            }
            if "_sb_metric" not in st.session_state or st.session_state["_sb_metric"] not in metric_options:
                st.session_state["_sb_metric"] = filters["metric"] if filters["metric"] in metric_options else "rmse"
            selected_metric = st.selectbox(
                "Primary Comparison Metric",
                options=metric_options,
                format_func=lambda m: metric_labels.get(m, m.upper()),
                key="_sb_metric",
            )
            filters["metric"] = selected_metric

        if has_any_filter:
            if st.button("Reset Filters to Default", width="stretch"):
                reset_global_filters()
                st.rerun()

        st.markdown("---")
        st.markdown("### Research Integrity Protocol")
        st.markdown(
            f"""
            <div style="font-size: 0.74rem; color: #94a3b8; line-height: 1.45;">
                <div style="margin-bottom: 5px;"><strong style="color: #00d2c4;">• Chronological Splits:</strong> {DATASET_SPLITS['train']:,} Train | {DATASET_SPLITS['validation']:,} Val | {DATASET_SPLITS['conformal']:,} Cal | {DATASET_SPLITS['test']:,} Test.</div>
                <div style="margin-bottom: 5px;"><strong style="color: #00d2c4;">• Artifact Immutability:</strong> Read-only access to frozen checkpoints &amp; parquet results.</div>
                <div style="margin-bottom: 5px;"><strong style="color: #00d2c4;">• Ground Truth:</strong> BAM-1020 federal reference monitors (zero input leakage).</div>
                <div><strong style="color: #00d2c4;">• Execution Mode:</strong> Local CPU &amp; single-node Spark/DuckDB runtime.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return filters.copy()


def render_sidebar(
    show_site_filter: bool = True,
    show_date_filter: bool = True,
    show_scenario_filter: bool = False,
    show_pipeline_filter: bool = False,
    show_population_filter: bool = False,
    show_metric_filter: bool = False,
    active_page: Optional[str] = None,
) -> Tuple[str, Optional[datetime.date], Optional[datetime.date]]:
    """
    Backward-compatible sidebar renderer returning (selected_site, start_date, end_date).
    All extended filter selections are synchronized in st.session_state['airwise_filters'].
    """
    filters = _render_sidebar_controls(
        show_site_filter=show_site_filter,
        show_date_filter=show_date_filter,
        show_scenario_filter=show_scenario_filter,
        show_pipeline_filter=show_pipeline_filter,
        show_population_filter=show_population_filter,
        show_metric_filter=show_metric_filter,
        active_page=active_page,
    )
    return filters["site"], filters["start_date"], filters["end_date"]


def render_benchmark_sidebar(
    show_site_filter: bool = True,
    show_date_filter: bool = False,
    show_scenario_filter: bool = True,
    show_pipeline_filter: bool = True,
    show_population_filter: bool = True,
    show_metric_filter: bool = True,
    active_page: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extended sidebar renderer returning the full global filter state dictionary:
    {'site', 'start_date', 'end_date', 'scenario_id', 'pipeline_code', 'population_mode', 'metric'}.
    """
    return _render_sidebar_controls(
        show_site_filter=show_site_filter,
        show_date_filter=show_date_filter,
        show_scenario_filter=show_scenario_filter,
        show_pipeline_filter=show_pipeline_filter,
        show_population_filter=show_population_filter,
        show_metric_filter=show_metric_filter,
        active_page=active_page,
    )

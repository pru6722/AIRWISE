"""
AIRWISE Dashboard: Architectural & Methodology Diagrams
Renders polished, dark-themed HTML/CSS visual schematics for:
1. Overall End-to-End Research Workflow
2. Four Comparative Calibration & Reconstruction Pipelines (A, B, C, D)
3. Joint Causal TCN v2 Architecture (ChannelLayerNorm1d, RF=61h)
4. 4-Way Chronological Data Partition Protocol
"""

from typing import Optional
import streamlit as st
from dashboard.config import THEME, DATASET_SPLITS, PIPELINES


def _arrow_right(color: str = "#00d2c4") -> str:
    return (
        f'<div style="display:flex;align-items:center;justify-content:center;'
        f'color:{color};font-weight:800;font-size:1.1rem;padding:0 2px;">&#8594;</div>'
    )


def _arrow_down(color: str = "#00d2c4") -> str:
    return (
        f'<div style="text-align:center;color:{color};font-weight:800;'
        f'font-size:1.1rem;line-height:1.1;margin:4px 0;">&#8595;</div>'
    )


def render_research_workflow_diagram() -> None:
    """Renders the 7-stage end-to-end AIRWISE research methodology workflow."""
    stages = [
        ("1. Raw Sensors", "PurpleAir + BAM-1020<br/>18,584 Hourly Obs<br/>3 Indian Cities", "#00a8ff"),
        ("2. Preprocessing", "Gap Segmentation<br/>Chronological 60/10/10/20<br/>Dew Point Excluded", "#00d2c4"),
        ("3. Four Pipelines", "A: Imp&#8594;Cal | B: Cal&#8594;Imp<br/>C: Joint TCN v2<br/>D: Complete Case", "#bd93f9"),
        ("4. 9 Scenarios", "MCAR, MAR-RH, MAR-Lag<br/>Blocks (1h–24h Nom.)<br/>MNAR-PM Stress", "#ffb86c"),
        ("5. Bootstrap", "Moving-Block (b=24h)<br/>95% Paired CIs &amp;<br/>Sensitivity Sweep", "#00d2c4"),
        ("6. Uncertainty", "Inductive Split Conformal<br/>Regime &amp; Block Intervals<br/>Site Coverage Audit", "#00a8ff"),
        ("7. Regulatory & Big Data", "24h CPCB Reliability<br/>75% Rule | Spark/DuckDB<br/>CSV vs Parquet vs ORC", "#bd93f9"),
    ]

    cards_html = []
    for idx, (title, body, accent) in enumerate(stages):
        cards_html.append(
            f'<div style="flex:1;min-width:130px;background-color:#131c2e;'
            f'border:1px solid #1e293b;border-top:3px solid {accent};border-radius:8px;'
            f'padding:10px 10px;text-align:center;">'
            f'<div style="font-size:0.78rem;font-weight:700;color:{accent};margin-bottom:4px;">{title}</div>'
            f'<div style="font-size:0.71rem;color:#cbd5e1;line-height:1.35;">{body}</div>'
            f'</div>'
        )
        if idx < len(stages) - 1:
            cards_html.append(_arrow_right(accent))

    html = (
        '<div style="display:flex;flex-wrap:wrap;align-items:stretch;gap:6px;'
        'padding:12px;background-color:#0e1626;border:1px solid #1e293b;border-radius:10px;">'
        + "".join(cards_html)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_four_pipelines_diagram(active_pipeline: Optional[str] = None) -> None:
    """Renders side-by-side structural comparison of Pipelines A, B, C, and D."""
    specs = [
        ("A", "Stage 1: KNN / Iter Imputer (Train-Only)", "Stage 2: RF Calibrator on Imputed Inputs", "Full Test Coverage (100%)", "Sensitive to missing channel uncertainty before calibration"),
        ("B", "Stage 1: RF Calibrator on Observed Data", "Stage 2: Temporal Imputation of Missing States", "Full Test Coverage (100%)", "Lowest Full-Test RMSE across all 9 missingness scenarios"),
        ("C", "Shared Encoder: Dilated Causal TCN v2", "Dual Heads: Mask-Gated Calibration & Recon", "Full Test Coverage (100%)", "Strong on short 1h–3h gaps; compounds on multi-day outages"),
        ("D", "Filter: Strict Complete Cases (m_t = 1)", "Single Stage: RF Calibrator (No Imputation)", "Partial Coverage (e.g. 79.3%)", "Pure calibration ceiling reference; excludes all missing hours"),
    ]

    cols_html = []
    for code, s1, s2, cov, note in specs:
        p = PIPELINES[code]
        is_active = active_pipeline == code
        border = f"2px solid {p['color']}" if is_active else "1px solid #1e293b"
        shadow = f"0 0 14px rgba(0, 210, 196, 0.18)" if is_active else "none"
        badge = (
            f'<span style="float:right;font-size:0.68rem;background:{p["color"]};'
            f'color:#0b111e;padding:1px 6px;border-radius:4px;font-weight:700;">ACTIVE</span>'
            if is_active
            else ""
        )

        card = (
            f'<div style="flex:1;min-width:210px;background-color:#131c2e;border:{border};'
            f'box-shadow:{shadow};border-radius:8px;padding:12px;">'
            f'<div style="font-size:0.86rem;font-weight:700;color:{p["color"]};margin-bottom:8px;">'
            f'{p["name"]}{badge}</div>'
            f'<div style="background:#0e1626;border:1px solid #1e293b;border-radius:5px;padding:7px;font-size:0.75rem;color:#f8fafd;">{s1}</div>'
            + _arrow_down(p["color"])
            + f'<div style="background:#0e1626;border:1px solid #1e293b;border-radius:5px;padding:7px;font-size:0.75rem;color:#f8fafd;">{s2}</div>'
            + _arrow_down(p["color"])
            + f'<div style="background:rgba(19,28,46,0.9);border:1px dashed {p["color"]};border-radius:5px;padding:6px;font-size:0.74rem;font-weight:600;color:{p["color"]};text-align:center;">{cov}</div>'
            f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:8px;line-height:1.35;">{note}</div>'
            f'</div>'
        )
        cols_html.append(card)

    html = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 14px 0;">' + "".join(cols_html) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_joint_tcn_diagram() -> None:
    """Renders polished dark-themed layered schematic of Joint Causal TCN v2."""
    html = (
        '<div style="background:#0e1626;border:1px solid #1e293b;border-radius:10px;padding:14px;">'
        '<div style="font-size:0.82rem;font-weight:700;color:#bd93f9;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.6px;">'
        'Joint Causal TCN v2 Architecture (Kernel = 3 | Dilations = [1, 2, 4, 8] | RF = 61h | Seq Context = 64h)'
        '</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px;">'
        '<div style="flex:1;min-width:170px;background:#131c2e;border:1px solid #00a8ff;border-radius:6px;padding:8px;">'
        '<div style="font-size:0.76rem;font-weight:700;color:#00a8ff;">Raw PurpleAir (3)</div>'
        '<div style="font-size:0.72rem;color:#cbd5e1;">PA_CF1, PA_ATM, PA_ALT (Masked when M_t = 0)</div></div>'
        '<div style="flex:1;min-width:170px;background:#131c2e;border:1px solid #00d2c4;border-radius:6px;padding:8px;">'
        '<div style="font-size:0.76rem;font-weight:700;color:#00d2c4;">Meteo &amp; Temporal (7)</div>'
        '<div style="font-size:0.72rem;color:#cbd5e1;">RH, Temp, hour(sin/cos), doy(sin/cos), dt_hours</div></div>'
        '<div style="flex:1;min-width:170px;background:#131c2e;border:1px solid #ffb86c;border-radius:6px;padding:8px;">'
        '<div style="font-size:0.76rem;font-weight:700;color:#ffb86c;">Availability Indicator (1)</div>'
        '<div style="font-size:0.72rem;color:#cbd5e1;">Binary Mask M_t in {0, 1} (Zero BAM Input)</div></div>'
        '</div>'
        + _arrow_down("#00a8ff")
        + '<div style="background:#131c2e;border:1px solid #bd93f9;border-radius:8px;padding:10px;margin:4px 0;">'
        '<div style="font-size:0.78rem;font-weight:700;color:#bd93f9;text-align:center;margin-bottom:6px;">'
        'Shared Causal Dilated Temporal Encoder (ChannelLayerNorm1d | 144 Causality Checks: Max Dev = 0.0000)'
        '</div>'
        '<div style="display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:6px;">'
        '<span style="background:#0e1626;border:1px solid #334155;border-radius:4px;padding:4px 8px;font-size:0.72rem;color:#f8fafd;">1x1 Causal Proj (11&#8594;32)</span>'
        '<span style="color:#bd93f9;">&#8594;</span>'
        '<span style="background:#0e1626;border:1px solid #334155;border-radius:4px;padding:4px 8px;font-size:0.72rem;color:#f8fafd;">Block 1 (d=1, K=3)</span>'
        '<span style="color:#bd93f9;">&#8594;</span>'
        '<span style="background:#0e1626;border:1px solid #334155;border-radius:4px;padding:4px 8px;font-size:0.72rem;color:#f8fafd;">Block 2 (d=2, K=3)</span>'
        '<span style="color:#bd93f9;">&#8594;</span>'
        '<span style="background:#0e1626;border:1px solid #334155;border-radius:4px;padding:4px 8px;font-size:0.72rem;color:#f8fafd;">Block 3 (d=4, K=3)</span>'
        '<span style="color:#bd93f9;">&#8594;</span>'
        '<span style="background:#0e1626;border:1px solid #334155;border-radius:4px;padding:4px 8px;font-size:0.72rem;color:#f8fafd;">Block 4 (d=8, K=3)</span>'
        '</div></div>'
        + _arrow_down("#bd93f9")
        + '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:4px 0;">'
        '<div style="flex:1;min-width:220px;background:#131c2e;border:1px solid #00d2c4;border-radius:6px;padding:9px;">'
        '<div style="font-size:0.78rem;font-weight:700;color:#00d2c4;">Calibration Head (M_t = 1)</div>'
        '<div style="font-size:0.72rem;color:#cbd5e1;">MLP(32 &#8594; 16 &#8594; 1) &#8594; y_cal<br/>Supervised by Huber Loss when PM observed</div></div>'
        '<div style="flex:1;min-width:220px;background:#131c2e;border:1px solid #ffb86c;border-radius:6px;padding:9px;">'
        '<div style="font-size:0.78rem;font-weight:700;color:#ffb86c;">Reconstruction Head (M_t = 0)</div>'
        '<div style="font-size:0.72rem;color:#cbd5e1;">MLP([H_t, y_cal * M_t, M_t] &#8594; 16 &#8594; 1) &#8594; y_rec<br/>Unmasked estimate strictly zeroed during outage</div></div>'
        '</div>'
        + _arrow_down("#00d2c4")
        + '<div style="background:#131c2e;border:1px solid #00d2c4;border-radius:6px;padding:8px;text-align:center;">'
        '<span style="font-size:0.8rem;font-weight:700;color:#00d2c4;">Composite Output: </span>'
        '<code style="color:#f8fafd;font-size:0.78rem;">y_hat_t = M_t * y_cal + (1 - M_t) * y_rec</code>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_conformal_split_diagram() -> None:
    """Renders polished 4-way chronological partition schematic with reconciled sample counts."""
    splits = [
        ("Training Split (60.0%)", f"N = {DATASET_SPLITS['train']:,} Rows", "BLR: 4,464 | DEL: 4,502 | HAM: 2,183", "Fits Imputers, Calibrators & Joint TCN v2", "#00a8ff"),
        ("Validation Split (10.0%)", f"N = {DATASET_SPLITS['validation']:,} Rows", "BLR: 744 | DEL: 750 | HAM: 363", "Early Stopping (Epoch 6) & Hyperparameters", "#bd93f9"),
        ("Conformal Calib (10.0%)", f"N = {DATASET_SPLITS['conformal']:,} Rows", "BLR: 744 | DEL: 750 | HAM: 364", "Nonconformity Residuals |Y - Y_hat| & Quantiles", "#ffb86c"),
        ("Independent Test (20.0%)", f"N = {DATASET_SPLITS['test']:,} Rows", "BLR: 1,489 | DEL: 1,502 | HAM: 729", "Frozen 9-Scenario Benchmark & Coverage Audit", "#00d2c4"),
    ]

    cards = []
    for title, n_str, site_str, role, color in splits:
        cards.append(
            f'<div style="flex:1;min-width:190px;background:#131c2e;border:1px solid #1e293b;'
            f'border-left:4px solid {color};border-radius:6px;padding:10px;">'
            f'<div style="font-size:0.78rem;font-weight:700;color:{color};">{title}</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:#f8fafd;margin:2px 0;">{n_str}</div>'
            f'<div style="font-size:0.69rem;color:#94a3b8;margin-bottom:4px;">{site_str}</div>'
            f'<div style="font-size:0.73rem;color:#cbd5e1;line-height:1.3;">{role}</div>'
            f'</div>'
        )

    html = (
        '<div style="background:#0e1626;border:1px solid #1e293b;border-radius:10px;padding:12px;">'
        f'<div style="font-size:0.8rem;font-weight:700;color:#f8fafd;margin-bottom:8px;">'
        f'Strictly Chronological Per-Site Partitioning (Total N = {DATASET_SPLITS["total"]:,} Hours — Zero Temporal Leakage)</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
        + "".join(cards)
        + '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

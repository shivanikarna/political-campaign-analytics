"""Look-and-feel helpers (CSS, hero banner, consistent Plotly styling)."""
from __future__ import annotations

import html
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1400px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(120deg,#0F172A 0%,#1E3A8A 55%,#7F1D1D 100%);
       border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 18px;
       box-shadow: 0 10px 30px rgba(15,23,42,.25);}
.hero h1 {color:#fff; margin:0 0 4px 0; font-size: 1.9rem; letter-spacing:-.02em;}
.hero p {color:#CBD5E1; margin:0 0 12px 0; font-size:.98rem;}
.badge {display:inline-block; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
        color:#F1F5F9; padding:3px 11px; border-radius:999px; font-size:.78rem; margin:0 6px 4px 0;}
[data-testid="stMetric"] {background:#fff; border:1px solid #E2E8F0; border-radius:14px;
        padding:14px 16px; box-shadow:0 1px 2px rgba(15,23,42,.05); transition:all .15s ease;}
[data-testid="stMetric"]:hover {box-shadow:0 8px 20px rgba(15,23,42,.10); transform:translateY(-2px);
        border-color:#C7D2FE;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {white-space: normal !important;
        overflow: visible !important; text-overflow: clip !important;}
[data-testid="stMetricLabel"] p {font-size:.74rem; color:#64748B; font-weight:600;
        text-transform:uppercase; letter-spacing:.02em; line-height:1.25;}
[data-testid="stMetricValue"] {font-weight:700; letter-spacing:-.02em;}
.stTabs [data-baseweb="tab-list"] {gap:4px; border-bottom:1px solid #E2E8F0;}
.stTabs [data-baseweb="tab"] {border-radius:10px 10px 0 0; padding:10px 16px; font-weight:600;}
.stTabs [aria-selected="true"] {background:#EEF2FF; color:#4338CA;}
.section-title {font-size:1.12rem; font-weight:700; margin:1.1rem 0 .1rem 0; color:#0F172A;}
.section-sub {color:#64748B; font-size:.88rem; margin-bottom:.6rem;}
.insight {background:#F8FAFF; border-left:4px solid #6366F1; padding:10px 14px;
          border-radius:0 10px 10px 0; margin:6px 0; color:#1E293B;}
section[data-testid="stSidebar"] {background:#F1F5F9;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: list[str]) -> None:
    chips = "".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges)
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{chips}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def insight(text: str) -> None:
    """Callout box; **bold** markers in `text` are converted to <b> (HTML-escaped first)."""
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))
    st.markdown(f'<div class="insight">{safe}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 400, legend: str = "top") -> go.Figure:
    legends = {
        "top": dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        "right": dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, title=None),
    }
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=75 if legend == "top" else 60, b=10),
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=13, color="#0F172A"),
        title=dict(font=dict(size=15), x=0, xanchor="left", y=0.97, yanchor="top", yref="container"),
        hoverlabel=dict(bgcolor="white"),
    )
    if legend != "none":
        fig.update_layout(legend=legends[legend])
    else:
        fig.update_layout(showlegend=False)
    return fig


def show(fig: go.Figure, height: int = 400, legend: str = "top") -> None:
    """Render a Plotly figure full-width (compatible across Streamlit versions)."""
    style_fig(fig, height, legend)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def download(label: str, data: bytes, file_name: str, mime: str, key: str, primary: bool = False) -> None:
    kwargs = dict(file_name=file_name, mime=mime, key=key, type="primary" if primary else "secondary")
    try:
        st.download_button(label, data, width="stretch", **kwargs)
    except TypeError:
        st.download_button(label, data, use_container_width=True, **kwargs)


def table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)

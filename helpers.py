"""
Shared helpers for Streamlit pages — theme CSS & cached data loaders.
"""
import streamlit as st
import sys, os

# Ensure project root on path so pages/ can import this and analysis
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis import (
    load_data, monthly_resample, descriptive_stats,
    decompose_series, run_all_models, best_model, split_data,
)

# ──────────────────────────────────────────
# CACHED DATA
# ──────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_data()
    monthly = monthly_resample(df)
    return df, monthly

@st.cache_data
def get_models(_test_months: int = 6):
    _, monthly = get_data()
    train, test, results, metrics_df, model_params = run_all_models(monthly, _test_months)
    return train, test, results, metrics_df, model_params


@st.cache_data
def get_models_compare(_test_months: int = 6, _short_years: int = 5):
    """Compare: full 10-yr data vs last N years — same test period."""
    _, monthly = get_data()
    full = run_all_models(monthly, _test_months)
    cutoff_year = monthly.index[-1].year - _short_years + 1
    monthly_short = monthly[monthly.index.year >= cutoff_year].copy()
    short = run_all_models(monthly_short, _test_months)
    return full, short, len(monthly), len(monthly_short)

# ──────────────────────────────────────────
# THEME CSS (inject on every page)
# ──────────────────────────────────────────
THEME_CSS = """
<style>
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }
    .page-title {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 700; padding: 0.5rem 0 0.2rem 0;
    }
    .section-box {
        background: linear-gradient(135deg, rgba(255,215,0,0.06), rgba(255,165,0,0.02));
        border: 1px solid rgba(255,215,0,0.15); border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0;
    }
    .highlight-box {
        background: rgba(78,205,196,0.08); border-left: 4px solid #4ECDC4;
        border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 0.6rem 0;
    }
    .warn-box {
        background: rgba(255,107,107,0.08); border-left: 4px solid #FF6B6B;
        border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 0.6rem 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,215,0,0.06); border: 1px solid rgba(255,215,0,0.15);
        border-radius: 10px; padding: 0.8rem;
    }
    .stSidebar > div:first-child { background: linear-gradient(180deg, #0d1117, #161b22); }
</style>
"""

def inject_css():
    st.markdown(THEME_CSS, unsafe_allow_html=True)

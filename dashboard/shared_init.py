"""
Shared Initialization Module
Single source of truth for loading data and building the shared engines.

Call ``ensure_initialized()`` at the top of the main app and inside the
``render()`` of every page. It is safe to call on every rerun: the heavy
objects are cached, so repeat calls are cheap no-ops.
"""
import streamlit as st
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_all_data, validate_data_quality, init_session_state
from src.risk_engine import create_risk_engine
from src.insight_generator import InsightGenerator
import src.kpi_calculator as kpi_calculator
from dashboard.components.insight_engine import create_insight_engine


# Keys that must exist in session state before any page renders.
REQUIRED_KEYS = ('data_dict', 'risk_engine', 'insight_generator', 'insight_engine')


@st.cache_resource(show_spinner="Initializing ER Risk Dashboard...")
def _initialize_everything():
    """Load data and create all engines (cached per server process).

    Raises on failure so the caller can surface the real error to the user.
    """
    data_dict = load_all_data("data")

    quality = validate_data_quality(data_dict)
    if not quality.get('valid', False):
        raise RuntimeError(
            "Data validation failed: " + "; ".join(quality.get('issues', []) or ["unknown issue"])
        )

    risk_engine = create_risk_engine(
        data_dict['events'],
        data_dict['kpis'],
        data_dict['branches'],
        data_dict['employees'],
        data_dict.get('cases')
    )

    insight_generator = InsightGenerator(risk_engine, kpi_calculator)
    insight_engine = create_insight_engine(insight_generator)

    return data_dict, risk_engine, insight_generator, insight_engine, quality


def ensure_initialized():
    """Ensure shared data + engines exist in ``st.session_state``.

    Idempotent: runs the heavy init once per browser session, then becomes a
    cheap no-op. Always guarantees the base session defaults (filters,
    selected_branch, ...) exist. On failure the real error is surfaced and the
    app stops -- no placeholder/fake data is substituted.
    """
    # Always guarantee base defaults (filters, selected_branch, chat_history, ...)
    init_session_state()

    if all(k in st.session_state for k in REQUIRED_KEYS):
        return

    try:
        data_dict, risk_engine, insight_generator, insight_engine, quality = _initialize_everything()
    except Exception as exc:  # noqa: BLE001 - surface the real cause, then stop
        st.error(f"Failed to initialize dashboard data/engines: {exc}")
        st.stop()
        return

    st.session_state.data_dict = data_dict
    st.session_state.risk_engine = risk_engine
    st.session_state.insight_generator = insight_generator
    st.session_state.insight_engine = insight_engine
    st.session_state.data_quality = quality

    for warning in quality.get('warnings', []) or []:
        st.warning(f"Data Warning: {warning}")

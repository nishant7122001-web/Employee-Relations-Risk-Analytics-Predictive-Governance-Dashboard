"""
ER Risk Analytics & Predictive Governance Dashboard
Main Streamlit Application Entry Point
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.shared_init import ensure_initialized
from dashboard.components.chat_interface import render_sidebar_chat, render_chat_modal


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ER Risk Analytics | DCB Bank",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f4f8;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important;
        color: white !important;
    }
    div[data-testid="stSidebarNav"] {
        background: linear-gradient(180deg, #1f4e79 0%, #1a1a2e 100%);
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1f4e79;
    }
    .alert-critical { border-left-color: #dc3545; }
    .alert-high { border-left-color: #fd7e14; }
    .alert-medium { border-left-color: #ffc107; }
    .alert-low { border-left-color: #198754; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZATION
# ============================================================
# Single source of truth shared with every page (dashboard/shared_init.py).
# Loads data and builds engines once, then stores them in session state so all
# pages and reruns receive the same objects.
ensure_initialized()

data_dict = st.session_state.data_dict
risk_engine = st.session_state.risk_engine
insight_generator = st.session_state.insight_generator
insight_engine = st.session_state.insight_engine
quality = st.session_state.data_quality


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #1f4e79; margin: 0;">DCB Bank</h2>
        <p style="color: #6c757d; margin: 0;">ER Risk Analytics & Predictive Governance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Global filters
    from dashboard.components.filters import render_global_filters
    filters = render_global_filters(data_dict, "global", show_advanced=True)
    
    st.markdown("---")
    
    # Data freshness
    events_df = data_dict['events']
    if len(events_df) > 0:
        last_date = events_df['event_date'].max()
        st.caption(f"Data as of: {last_date.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # AI Assistant in sidebar
    render_sidebar_chat(
        insight_engine,
        insight_generator.build_ai_context(),
        current_branch=st.session_state.get('selected_branch'),
        current_category=st.session_state.get('selected_category')
    )
    
    st.markdown("---")
    
    # System info
    with st.expander("System Info"):
        st.write(f"**Records Loaded:**")
        for table, count in quality['record_counts'].items():
            st.write(f"  - {table}: {count:,}")
        st.write(f"**Models:** Attrition, Escalation, Forecasting")
        st.write(f"**Clustering:** DBSCAN Hotspot Detection")
        st.write(f"**AI Mode:** {insight_engine.mode}")


# ============================================================
# MAIN CONTENT - Page Navigation
# ============================================================
# Page titles for navigation (removed Case Management and Admin)
PAGES = {
    "1_Executive_Overview": "Executive Overview",
    "2_Branch_Deep_Dive": "Branch Deep-Dive",
    "3_Predictive_Analytics": "Predictive Analytics",
    "4_AI_Insights_Assistant": "AI Insights Assistant"
}

# Render page navigation
st.markdown("""
<nav style="margin-bottom: 1rem;">
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
""", unsafe_allow_html=True)

cols = st.columns(len(PAGES))
for i, (page_key, page_title) in enumerate(PAGES.items()):
    with cols[i]:
        if st.button(page_title, key=f"nav_{page_key}", width='stretch'):
            st.session_state.current_page = page_key
            st.rerun()

st.markdown("</div></nav>", unsafe_allow_html=True)

# Get current page (default to executive overview)
current_page = st.session_state.get('current_page', '1_Executive_Overview')

# Render the selected page
if current_page == '1_Executive_Overview':
    from dashboard.pages import _1_Executive_Overview
    _1_Executive_Overview.render()
elif current_page == '2_Branch_Deep_Dive':
    from dashboard.pages import _2_Branch_Deep_Dive
    _2_Branch_Deep_Dive.render()
elif current_page == '3_Predictive_Analytics':
    from dashboard.pages import _3_Predictive_Analytics
    _3_Predictive_Analytics.render()
elif current_page == '4_AI_Insights_Assistant':
    from dashboard.pages import _4_AI_Insights_Assistant
    _4_AI_Insights_Assistant.render()

# Render chat modal if open
render_chat_modal(insight_engine, insight_generator.build_ai_context())
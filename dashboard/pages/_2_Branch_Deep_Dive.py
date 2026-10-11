"""
Page 2: Branch Deep-Dive
Detailed branch-level risk analysis with department breakdown, tenure-risk matrix, and peer benchmarking
"""
import streamlit as st
import pandas as pd
import numpy as np

# Ensure session state is initialized (for direct page access)
from dashboard.shared_init import ensure_initialized

from dashboard.components.charts import (
    create_branch_risk_radar, create_department_breakdown,
    create_tenure_risk_matrix
)
from dashboard.components.kpi_cards import render_branch_kpi_cards, render_risk_gauge
from dashboard.components.filters import render_branch_selector, get_filtered_data
from dashboard.components.chat_interface import get_contextual_suggestions, render_suggestion_chips


def render():
    ensure_initialized()
    st.title("Branch Deep-Dive")
    st.caption("Granular branch-level risk analysis with department, tenure, and peer insights")
    
    # Get data from session state
    data_dict = st.session_state.data_dict
    risk_engine = st.session_state.risk_engine
    insight_generator = st.session_state.insight_generator
    
    # Apply filters
    filters = st.session_state.filters
    filtered_data = get_filtered_data(data_dict, filters)
    
    # Branch selector
    branches_df = filtered_data['branches'] if len(filtered_data['branches']) > 0 else data_dict['branches']
    branch_id = render_branch_selector(
        branches_df,
        key="branch_deep_dive_selector",
        label="Select Branch for Analysis"
    )
    
    if not branch_id:
        st.info("Please select a branch to begin analysis.")
        return
    
    # Store in session state for chat context
    st.session_state.selected_branch = branch_id
    
    # Get branch info
    branch_info = filtered_data['branches'][filtered_data['branches']['branch_id'] == branch_id]
    if len(branch_info) == 0:
        branch_info = data_dict['branches'][data_dict['branches']['branch_id'] == branch_id]
    
    branch_name = branch_info.iloc[0]['branch_name'] if len(branch_info) > 0 else branch_id
    branch_city = branch_info.iloc[0]['city'] if len(branch_info) > 0 else ''
    branch_region = branch_info.iloc[0]['region'] if len(branch_info) > 0 else ''
    
    st.markdown(f"### {branch_name} ({branch_city}, {branch_region})")
    
    # Generate branch insights
    with st.spinner("Analyzing branch..."):
        branch_insights = risk_engine.generate_branch_insights(branch_id)
    
    if not branch_insights or 'error' in branch_insights:
        st.error("Unable to generate insights for this branch.")
        return
    
    # ============================================================
    # RISK GAUGE + KPI CARDS
    # ============================================================
    col1, col2 = st.columns([1, 3])
    
    with col1:
        render_risk_gauge(
            branch_insights['risk_score'],
            f"{branch_name} Risk Score",
            size=250
        )
        
        # Convergence indicator
        conv = branch_insights['convergence_index']
        conv_color = "#dc3545" if conv > 0.7 else "#fd7e14" if conv > 0.5 else "#ffc107" if conv > 0.3 else "#198754"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: {conv_color}15; border-radius: 8px; border: 1px solid {conv_color}40;">
            <div style="font-size: 2rem; font-weight: 700; color: {conv_color};">{conv:.2f}</div>
            <div style="color: #6c757d; font-size: 0.85rem;">Convergence Index</div>
            <div style="font-size: 0.8rem; margin-top: 4px;">
                {'Critical' if conv > 0.7 else 'High' if conv > 0.5 else 'Moderate' if conv > 0.3 else 'Low'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        render_branch_kpi_cards(branch_insights)
    
    st.markdown("---")
    
    # ============================================================
    # TABS FOR DETAILED ANALYSIS
    # ============================================================
    tab1, tab2, tab3 = st.tabs([
        "Risk Radar", "Departments", "Tenure-Risk"
    ])
    
    with tab1:
        st.subheader("Risk Profile vs Industry Benchmarks")
        fig = create_branch_risk_radar(
            filtered_data['kpis'] if len(filtered_data['kpis']) > 0 else data_dict['kpis'],
            branch_id
        )
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("Department-Level ER Risk")
        fig = create_department_breakdown(
            filtered_data['employees'] if len(filtered_data['employees']) > 0 else data_dict['employees'],
            filtered_data['events'] if len(filtered_data['events']) > 0 else data_dict['events'],
            branch_id
        )
        st.plotly_chart(fig, width='stretch')
        
        # Department table
        depts = branch_insights.get('department_breakdown', [])
        if depts:
            df = pd.DataFrame(depts)
            df['avg_performance'] = df['avg_performance'].round(2)
            df['avg_tenure'] = df['avg_tenure'].round(1)
            df['high_performer_pct'] = (df['high_performer_pct'] * 100).round(0).astype(int)
            df.columns = ['Department', 'Headcount', 'Avg Performance', 'Avg Tenure (yrs)', 'High Performer %']
            st.dataframe(df, width='stretch', hide_index=True)
    
    with tab3:
        st.subheader("Employee Tenure vs Risk Matrix")
        fig = create_tenure_risk_matrix(
            filtered_data['employees'] if len(filtered_data['employees']) > 0 else data_dict['employees'],
            filtered_data['events'] if len(filtered_data['events']) > 0 else data_dict['events'],
            branch_id
        )
        st.plotly_chart(fig, width='stretch')
        
        st.caption("""
        **Quadrant Guide:**
        - **High Risk, High Tenure**: Tenured employees with issues - retention critical
        - **High Risk, Low Tenure**: New hires struggling - onboarding/manager issue
        - **Low Risk, High Tenure**: Stable core performers
        - **Low Risk, Low Tenure**: New hires performing well
        """)
    
    # ============================================================
    # RECENT EVENTS
    # ============================================================
    st.markdown("---")
    st.subheader("Recent Events (Last 90 Days)")
    
    recent = branch_insights.get('recent_events', [])
    if recent:
        df = pd.DataFrame(recent)
        df.columns = ['Category', 'Count', 'Avg Severity', 'Escalated %']
        df['Avg Severity'] = df['Avg Severity'].round(1)
        df['Escalated %'] = (df['Escalated %'] * 100).round(1)
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("No recent events for this branch.")
    
    # ============================================================
    # AI SUGGESTIONS
    # ============================================================
    st.markdown("---")
    suggestions = get_contextual_suggestions(
        insight_generator.build_ai_context(), branch_id=branch_id
    )
    render_suggestion_chips(suggestions, "branch", 
        lambda q: setattr(st.session_state, 'branch_pending_question', q))


if __name__ == "__main__":
    render()
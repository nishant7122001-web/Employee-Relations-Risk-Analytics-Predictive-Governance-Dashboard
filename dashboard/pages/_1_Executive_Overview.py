"""
Page 1: Executive Overview
Organization-wide ER risk dashboard with key metrics, trends, and alerts
"""
import streamlit as st
import pandas as pd
import numpy as np

# Ensure session state is initialized (for direct page access)
from dashboard.shared_init import ensure_initialized

from dashboard.components.charts import (
    create_risk_heatmap, create_risk_trend_chart, create_category_mix_chart,
    create_grievance_attrition_correlation
)
from dashboard.components.kpi_cards import render_executive_kpis, render_alert_card
from dashboard.components.filters import render_filter_summary, get_filtered_data
from dashboard.components.chat_interface import get_contextual_suggestions, render_suggestion_chips


def render():
    ensure_initialized()
    st.title("Executive Overview")
    st.caption("Organization-wide ER risk snapshot, trends, and priority alerts")
    
    # Get data from session state
    data_dict = st.session_state.data_dict
    risk_engine = st.session_state.risk_engine
    insight_generator = st.session_state.insight_generator
    insight_engine = st.session_state.insight_engine
    
    # Apply filters
    filters = st.session_state.filters
    filtered_data = get_filtered_data(data_dict, filters)
    
    # Render filter summary
    render_filter_summary(filters)
    
    # Generate insights
    with st.spinner("Generating insights..."):
        exec_insights = insight_generator.generate_executive_summary()
        org_insights = risk_engine.generate_org_insights()
    
    # ============================================================
    # TOP KPI ROW
    # ============================================================
    render_executive_kpis(exec_insights)
    
    st.markdown("---")
    
    # ============================================================
    # PREDICTIVE ALERTS
    # ============================================================
    alerts = exec_insights.get('predictive_alerts', [])
    if alerts:
        st.subheader("Predictive Alerts")
        for i, alert in enumerate(alerts):
            # Determine severity
            severity = 'medium'
            if 'HIGH PRIORITY' in alert or 'CRITICAL' in alert or 'ESCALATION' in alert:
                severity = 'high'
            if 'URGENT' in alert or 'CONVERGENCE' in alert:
                severity = 'critical'
            
            render_alert_card({
                'severity': severity,
                'message': alert,
                'action': 'Review in Predictive Analytics page'
            }, i)
    
    st.markdown("---")
    
    # ============================================================
    # MAIN CHARTS ROW 1
    # ============================================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Branch Risk Heatmap")
        fig = create_risk_heatmap(
            filtered_data['branches'],
            filtered_data['kpis'],
            risk_engine
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("ER Category Mix")
        fig = create_category_mix_chart(filtered_data['events'])
        st.plotly_chart(fig, width='stretch')
    
    # ============================================================
    # MAIN CHARTS ROW 2
    # ============================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Trend")
        selected_branches = filters.get('branch_ids', [])
        fig = create_risk_trend_chart(
            filtered_data['kpis'],
            filtered_data['branches'],
            selected_branches if selected_branches else None
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Grievance vs Attrition Correlation")
        fig = create_grievance_attrition_correlation(filtered_data['kpis'])
        st.plotly_chart(fig, width='stretch')
    
    # ============================================================
    # TOP RISK BRANCHES TABLE
    # ============================================================
    st.markdown("---")
    st.subheader("Top Risk Branches")
    
    top_branches = exec_insights.get('top_risk_branches', [])
    if top_branches:
        df = pd.DataFrame(top_branches)
        df = df[['branch_id', 'branch_name', 'city', 'region', 'risk_score', 'convergence']]
        df.columns = ['Branch ID', 'Branch Name', 'City', 'Region', 'Risk Score', 'Convergence']
        df['Risk Score'] = df['Risk Score'].round(2)
        df['Convergence'] = df['Convergence'].round(3)
        
        # Color code risk
        def color_risk(val):
            if val >= 8.5: return 'background-color: #fef2f2'
            elif val >= 7: return 'background-color: #fff7ed'
            elif val >= 5: return 'background-color: #fffbeb'
            return ''
        
        styled = df.style.map(color_risk, subset=['Risk Score'])
        st.dataframe(styled, width='stretch', hide_index=True)
    else:
        st.info("No high-risk branches detected.")
    
    # ============================================================
    # SUGGESTED QUESTIONS
    # ============================================================
    st.markdown("---")
    suggestions = get_contextual_suggestions(
        {'executive_summary': exec_insights, 'org_insights': org_insights}
    )
    render_suggestion_chips(suggestions, "exec", 
        lambda q: setattr(st.session_state, 'exec_pending_question', q))


# For direct execution
if __name__ == "__main__":
    render()
"""
Page 3: Predictive Analytics
Attrition risk, escalation prediction, hotspot detection, forecasting, and what-if simulator
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure session state is initialized (for direct page access)
from dashboard.shared_init import ensure_initialized

from dashboard.components.charts import (
    create_attrition_risk_distribution, create_escalation_risk_scatter,
    create_hotspot_map, create_forecast_chart, create_what_if_chart
)
from dashboard.components.kpi_cards import render_kpi_card
from dashboard.components.filters import get_filtered_data
from dashboard.components.chat_interface import get_contextual_suggestions, render_suggestion_chips
from src.utils import apply_dashboard_theme


def render():
    ensure_initialized()
    st.title("Predictive Analytics")
    st.caption("ML-powered risk forecasting, attrition prediction, escalation risk, and scenario planning")
    
    # Get data from session state
    data_dict = st.session_state.data_dict
    risk_engine = st.session_state.risk_engine
    insight_generator = st.session_state.insight_generator
    
    # Apply filters
    filters = st.session_state.filters
    filtered_data = get_filtered_data(data_dict, filters)
    
    # ============================================================
    # TABS FOR PREDICTIVE MODULES
    # ============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Attrition Risk", "Escalation Risk", "Hotspots", "Forecasts", "What-If Simulator"
    ])
    
    # ---- TAB 1: ATTRITION RISK ----
    with tab1:
        st.subheader("Employee Attrition Risk Scoring")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            top_n = st.slider("Top N Employees", 10, 100, 50, key="attrition_top_n")
            risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05, key="attrition_threshold")
        
        with st.spinner("Scoring attrition risk..."):
            attrition_risks = risk_engine.predict_attrition_risk(top_n)
        
        if len(attrition_risks) > 0:
            # Filter by threshold
            risk_col = 'attrition_risk' if 'attrition_risk' in attrition_risks.columns else 'risk_score'
            high_risk = attrition_risks[attrition_risks[risk_col] >= risk_threshold]
            
            # KPI cards
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                render_kpi_card("Total Scored", len(attrition_risks), 'int', icon='')
            with kpi_col2:
                render_kpi_card("High Risk (≥70%)", 
                              len(attrition_risks[attrition_risks[risk_col] >= 0.7]), 'int', 
                              rating='Critical' if len(attrition_risks[attrition_risks[risk_col] >= 0.7]) > 10 else 'Poor',
                              icon='')
            with kpi_col3:
                render_kpi_card("Medium Risk (50-70%)", 
                              len(attrition_risks[(attrition_risks[risk_col] >= 0.5) & (attrition_risks[risk_col] < 0.7)]), 'int',
                              icon='')
            with kpi_col4:
                avg_risk = attrition_risks[risk_col].mean()
                render_kpi_card("Avg Risk Score", avg_risk, 'float2',
                              rating='Critical' if avg_risk > 0.5 else 'Poor' if avg_risk > 0.3 else 'Average',
                              icon='')
            
            # Distribution chart
            st.plotly_chart(
                create_attrition_risk_distribution(attrition_risks), 
                width='stretch'
            )
            
            # Top risk table
            st.markdown("### Top Attrition Risks")
            display_cols = ['employee_id', 'branch_id', 'department', 'grade', 
                          'performance_rating', 'tenure_years', risk_col]
            display_df = attrition_risks[display_cols].head(20).copy()
            display_df[risk_col] = (display_df[risk_col] * 100).round(1).astype(str) + '%'
            display_df.columns = ['Employee ID', 'Branch', 'Department', 'Grade', 
                                'Performance', 'Tenure (yrs)', 'Attrition Risk']
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            # Download
            csv = attrition_risks.to_csv(index=False)
            st.download_button(
                "Download Full Attrition Risk List",
                csv,
                "attrition_risk_scores.csv",
                "text/csv",
                width='stretch'
            )
        else:
            st.warning("Attrition model not trained or insufficient data.")
    
    # ---- TAB 2: ESCALATION RISK ----
    with tab2:
        st.subheader("Grievance Escalation Prediction")
        
        with st.spinner("Scoring escalation risk..."):
            esc_risks = risk_engine.predict_escalation_risk()
        
        if len(esc_risks) > 0:
            # KPI cards
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            
            with kpi_col1:
                high_esc = len(esc_risks[esc_risks['escalation_risk'] >= 0.7])
                render_kpi_card("High Escalation Risk (≥70%)", high_esc, 'int',
                              rating='Critical' if high_esc > 5 else 'Poor' if high_esc > 2 else 'Good',
                              icon='')
            with kpi_col2:
                med_esc = len(esc_risks[(esc_risks['escalation_risk'] >= 0.5) & (esc_risks['escalation_risk'] < 0.7)])
                render_kpi_card("Medium Risk (50-70%)", med_esc, 'int', icon='')
            with kpi_col3:
                avg_esc = esc_risks['escalation_risk'].mean()
                render_kpi_card("Avg Escalation Probability", avg_esc, 'float2',
                              icon='')
            
            # Scatter plot
            st.plotly_chart(
                create_escalation_risk_scatter(esc_risks),
                width='stretch'
            )
            
            # Table
            st.markdown("### Highest Escalation Risk Grievances")
            display_df = esc_risks[['event_id', 'branch_id', 'employee_id', 'severity', 'subtype', 'escalation_risk']].head(20).copy()
            display_df['escalation_risk'] = (display_df['escalation_risk'] * 100).round(1).astype(str) + '%'
            display_df.columns = ['Case ID', 'Branch', 'Employee', 'Severity', 'Type', 'Escalation Risk']
            st.dataframe(display_df, width='stretch', hide_index=True)
        else:
            st.warning("Escalation model not trained or insufficient grievance data.")
    
    # ---- TAB 3: HOTSPOTS ----
    with tab3:
        st.subheader("Risk Hotspot Detection (DBSCAN Clustering)")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            eps = st.slider("DBSCAN eps", 0.5, 3.0, 1.5, 0.1, key="hotspot_eps")
            min_samples = st.slider("Min Samples", 2, 10, 3, key="hotspot_min_samples")
        
        with st.spinner("Detecting hotspots..."):
            hotspots = risk_engine.detect_hotspots(min_samples=min_samples, eps=eps)
            hotspot_summary = risk_engine.get_hotspot_summary()
        
        if len(hotspot_summary) > 0:
            # Cluster summary cards
            st.markdown("### Detected Clusters")
            for _, cluster in hotspot_summary.iterrows():
                with st.expander(f"Cluster {cluster['cluster']} - {cluster['branch_count']} branches (Avg Risk: {cluster['avg_risk_score']:.1f})"):
                    st.write(f"**Regions:** {cluster['regions']}")
                    st.write(f"**Branches:** {cluster['top_branches']}")
                    
                    # Get branch details for this cluster
                    cluster_branches = hotspots[hotspots['cluster'] == cluster['cluster']]
                    st.dataframe(
                        cluster_branches[['branch_name', 'city', 'region', 'risk_score']].sort_values('risk_score', ascending=False),
                        width='stretch', hide_index=True
                    )
            
            # Visualization
            st.plotly_chart(create_hotspot_map(hotspot_summary), width='stretch')
        else:
            st.info("No significant hotspot clusters detected with current parameters.")
            st.caption("Try adjusting eps (neighborhood radius) or min_samples.")
    
    # ---- TAB 4: FORECASTS ----
    with tab4:
        st.subheader("Risk Forecasting (Prophet Time-Series)")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            forecast_branch = st.selectbox(
                "Select Branch (or Org-wide)",
                ['Organization'] + data_dict['branches']['branch_name'].tolist(),
                key="forecast_branch"
            )
            forecast_horizon = st.slider("Forecast Horizon (months)", 1, 6, 3, key="forecast_horizon")
        
        if forecast_branch == 'Organization':
            with st.spinner("Generating organization forecast..."):
                org_forecast = risk_engine.forecast_org_risk(forecast_horizon)
                # Get historical
                org_kpis = filtered_data['kpis'].groupby('month').agg({
                    'grievance_rate': 'mean', 'disciplinary_rate': 'mean',
                    'absenteeism_rate': 'mean', 'attrition_rate_annualized': 'mean',
                    'whistleblower_rate': 'mean', 'grievance_escalation_rate': 'mean'
                }).reset_index()
                if 'risk_score' not in org_kpis.columns:
                    org_kpis['risk_score'] = org_kpis.apply(risk_engine.compute_branch_risk_score, axis=1)
            
            fig = create_forecast_chart(
                org_kpis, org_forecast, 'risk_score',
                "Organization Risk Score Forecast"
            )
            st.plotly_chart(fig, width='stretch')
            
            # Category forecasts
            st.markdown("### Category-Level Forecasts")
            cat_cols = st.columns(3)
            categories = [
                ('grievance_rate', 'Grievance Rate'),
                ('disciplinary_rate', 'Disciplinary Rate'),
                ('absenteeism_rate', 'Absenteeism Rate'),
                ('attrition_rate_annualized', 'Attrition Rate'),
                ('whistleblower_rate', 'Whistleblower Rate'),
                ('grievance_escalation_rate', 'Escalation Rate')
            ]
            
            for i, (col, name) in enumerate(categories):
                with cat_cols[i % 3]:
                    if col in org_kpis.columns:
                        try:
                            # Simple linear forecast for categories
                            y = org_kpis[col].values
                            x = np.arange(len(y))
                            slope, intercept = np.polyfit(x, y, 1)
                            forecast_vals = [intercept + slope * (len(y) + j) for j in range(1, forecast_horizon+1)]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=org_kpis['month'], y=y, name='Historical', line=dict(color='#1f4e79')))
                            fig.add_trace(go.Scatter(x=list(range(org_kpis['month'].max()+1, org_kpis['month'].max()+forecast_horizon+1)), 
                                                   y=forecast_vals, name='Forecast', line=dict(color='#fd7e14', dash='dash')))
                            fig.update_layout(title=name, height=250, showlegend=False)
                            fig = apply_dashboard_theme(fig, height=250)
                            st.plotly_chart(fig, width='stretch')
                        except Exception:
                            pass
        else:
            # Branch-specific forecast
            branch_id = data_dict['branches'][data_dict['branches']['branch_name'] == forecast_branch]['branch_id'].values[0]
            
            st.markdown(f"### {forecast_branch} Category Forecasts")
            cat_cols = st.columns(2)
            categories = ['Grievance', 'Disciplinary', 'Absenteeism', 'Attrition', 'Whistleblower']
            
            for i, cat in enumerate(categories):
                with cat_cols[i % 2]:
                    with st.spinner(f"Forecasting {cat}..."):
                        forecast = risk_engine.forecast_branch_category(branch_id, cat, forecast_horizon)
                    
                    if len(forecast) > 0:
                        # Get historical
                        branch_kpis = filtered_data['kpis'][filtered_data['kpis']['branch_id'] == branch_id].sort_values('month')
                        col_map = {
                            'Grievance': 'grievance_rate', 'Disciplinary': 'disciplinary_rate',
                            'Absenteeism': 'absenteeism_rate', 'Attrition': 'attrition_rate_annualized',
                            'Whistleblower': 'whistleblower_rate'
                        }
                        hist_col = col_map.get(cat, 'grievance_rate')
                        
                        fig = create_forecast_chart(
                            branch_kpis, forecast, hist_col,
                            f"{forecast_branch} - {cat} Forecast"
                        )
                        st.plotly_chart(fig, width='stretch')
    
    # ---- TAB 5: WHAT-IF SIMULATOR ----
    with tab5:
        st.subheader("Intervention Impact Simulator")
        st.caption("Model the effect of targeted interventions on branch risk scores")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            sim_branch = st.selectbox(
                "Target Branch",
                data_dict['branches']['branch_name'].tolist(),
                key="sim_branch"
            )
            
            intervention = st.selectbox(
                "Intervention Type",
                [
                    'grievance_mediation',
                    'manager_training',
                    'wellbeing_program',
                    'compensation_review',
                    'policy_reinforcement',
                    'whistleblower_protection'
                ],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="sim_intervention"
            )
            
            intensity = st.slider(
                "Implementation Intensity",
                0.1, 1.0, 0.5, 0.1,
                help="0.1 = Pilot, 0.5 = Partial rollout, 1.0 = Full implementation",
                key="sim_intensity"
            )
            
            if st.button("Run Simulation", type="primary", width='stretch'):
                branch_id = data_dict['branches'][data_dict['branches']['branch_name'] == sim_branch]['branch_id'].values[0]
                
                with st.spinner("Simulating intervention impact..."):
                    result = risk_engine.simulate_intervention(branch_id, intervention, intensity)
                
                if result:
                    st.session_state.sim_result = result
        
        with col2:
            if 'sim_result' in st.session_state:
                result = st.session_state.sim_result
                
                # Result cards
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                
                with kpi_col1:
                    render_kpi_card("Current Risk", result['current_risk'], 'float1', icon='')
                with kpi_col2:
                    render_kpi_card("Projected Risk", result['projected_risk'], 'float1', 
                                  color="#198754" if result['risk_reduction'] > 0 else "#dc3545", icon='')
                with kpi_col3:
                    render_kpi_card("Risk Reduction", result['risk_reduction'], 'float1',
                                  rating='Excellent' if result['risk_reduction'] > 1 else 'Good' if result['risk_reduction'] > 0.5 else 'Average',
                                  icon='')
                
                st.markdown(f"""
                **Intervention:** {result['intervention'].replace('_', ' ').title()}  
                **Intensity:** {result['intensity']*100:.0f}%  
                **Projected Reduction:** {result['reduction_pct']:.1f}%
                """)
                
                # Multi-intervention comparison
                st.markdown("---")
                st.markdown("### Compare All Interventions")
                
                all_interventions = [
                    'grievance_mediation', 'manager_training', 'wellbeing_program',
                    'compensation_review', 'policy_reinforcement', 'whistleblower_protection'
                ]
                
                results = []
                for intr in all_interventions:
                    r = risk_engine.simulate_intervention(
                        data_dict['branches'][data_dict['branches']['branch_name'] == sim_branch]['branch_id'].values[0],
                        intr, intensity
                    )
                    if r:
                        results.append(r)
                
                if results:
                    fig = create_what_if_chart(results)
                    st.plotly_chart(fig, width='stretch')
    
    # ============================================================
    # AI SUGGESTIONS
    # ============================================================
    st.markdown("---")
    suggestions = get_contextual_suggestions(
        insight_generator.build_ai_context(), category="predictive"
    )
    render_suggestion_chips(suggestions, "predictive", 
        lambda q: setattr(st.session_state, 'predictive_pending_question', q))


if __name__ == "__main__":
    render()
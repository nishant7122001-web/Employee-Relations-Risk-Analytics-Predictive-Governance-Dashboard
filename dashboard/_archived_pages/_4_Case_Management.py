"""
Page 4: Case Management
Case lifecycle, SLA tracking, investigator workload, resolution analytics
"""
import streamlit as st
import pandas as pd
import numpy as np

# Ensure session state is initialized (for direct page access)
from dashboard.shared_init import ensure_initialized

from dashboard.components.charts import (
    create_case_funnel, create_sla_breach_chart,
    create_resolution_time_box, create_investigator_workload
)
from dashboard.components.kpi_cards import render_kpi_card, render_kpi_table
from dashboard.components.filters import get_filtered_data, render_category_tabs
from dashboard.components.chat_interface import get_contextual_suggestions, render_suggestion_chips
from src.utils import format_number
from src.kpi_calculator import calculate_all_kpis


def render():
    ensure_initialized()
    st.title("📋 Case Management")
    st.caption("ER case lifecycle, SLA compliance, resolution analytics, and investigator workload")
    
    # Get data from session state
    data_dict = st.session_state.data_dict
    risk_engine = st.session_state.risk_engine
    insight_generator = st.session_state.insight_generator
    
    # Apply filters
    filters = st.session_state.filters
    filtered_data = get_filtered_data(data_dict, filters)
    
    cases_df = filtered_data.get('cases', pd.DataFrame())
    events_df = filtered_data.get('events', pd.DataFrame())
    employees_df = filtered_data.get('employees', pd.DataFrame())
    branches_df = filtered_data.get('branches', pd.DataFrame())
    
    # Cases don't carry employee_id - enrich from events so search/details work
    if (len(cases_df) > 0 and 'employee_id' not in cases_df.columns
            and 'event_id' in cases_df.columns and 'employee_id' in events_df.columns):
        cases_df = cases_df.merge(
            events_df[['event_id', 'employee_id']].drop_duplicates('event_id'),
            on='event_id', how='left'
        )
    
    if len(cases_df) == 0:
        st.warning("No case data available for current filters.")
        return
    
    # ============================================================
    # OVERVIEW KPIs
    # ============================================================
    st.subheader("📊 Case Overview")
    
    # Calculate case KPIs
    total_cases = len(cases_df)
    open_cases = len(cases_df[cases_df['status'].isin(['Open', 'In Progress', 'Under Review'])])
    resolved_cases = len(cases_df[cases_df['status'].isin(['Resolved', 'Closed'])])
    sla_breaches = cases_df['sla_breach'].sum()
    sla_breach_rate = (sla_breaches / total_cases * 100) if total_cases > 0 else 0
    avg_days_open = cases_df['days_open'].mean()
    avg_cost = cases_df['actual_cost'].mean() if 'actual_cost' in cases_df.columns else 0
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    
    with kpi_col1:
        render_kpi_card("Total Cases", total_cases, 'int', icon='📋')
    with kpi_col2:
        render_kpi_card("Open Cases", open_cases, 'int', 
                      rating='Critical' if open_cases > 50 else 'Poor' if open_cases > 20 else 'Good',
                      icon='🔴')
    with kpi_col3:
        render_kpi_card("Resolved", resolved_cases, 'int', icon='✅')
    with kpi_col4:
        render_kpi_card("SLA Breach Rate", sla_breach_rate, 'float1',
                      rating='Critical' if sla_breach_rate > 20 else 'Poor' if sla_breach_rate > 10 else 'Good',
                      icon='⏰')
    with kpi_col5:
        render_kpi_card("Avg Days Open", avg_days_open, 'float1',
                      rating='Critical' if avg_days_open > 45 else 'Poor' if avg_days_open > 30 else 'Good',
                      icon='📅')
    with kpi_col6:
        render_kpi_card("Avg Cost", avg_cost, 'currency', icon='💰')
    
    st.markdown("---")
    
    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Lifecycle & Funnel", "⏱️ SLA & Resolution", "👮 Investigators", "📋 Case Explorer"
    ])
    
    # ---- TAB 1: LIFECYCLE ----
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Case Lifecycle Funnel")
            fig = create_case_funnel(cases_df)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Status Distribution")
            status_counts = cases_df['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            import plotly.express as px
            fig = px.pie(
                status_counts, values='Count', names='Status',
                color='Status',
                color_discrete_map={
                    'Open': '#dc3545', 'In Progress': '#fd7e14',
                    'Under Review': '#ffc107', 'Resolved': '#198754',
                    'Closed': '#6c757d', 'Escalated': '#6f42c1'
                },
                hole=0.4
            )
            from src.utils import apply_dashboard_theme
            fig = apply_dashboard_theme(fig, height=350)
            st.plotly_chart(fig, width='stretch')
        
        # Stage distribution by category
        st.subheader("Stage Distribution by Category")
        if 'stage' in cases_df.columns and 'category' in cases_df.columns:
            stage_cat = pd.crosstab(cases_df['category'], cases_df['stage'], margins=True)
            st.dataframe(stage_cat, width='stretch')
    
    # ---- TAB 2: SLA & RESOLUTION ----
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SLA Breach Rate by Category")
            fig = create_sla_breach_chart(cases_df)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Resolution Time Distribution")
            fig = create_resolution_time_box(cases_df)
            st.plotly_chart(fig, width='stretch')
        
        # SLA by branch
        st.subheader("SLA Performance by Branch")
        if 'branch_id' in cases_df.columns:
            branch_sla = cases_df.groupby('branch_id').agg(
                total_cases=('case_id', 'count'),
                breaches=('sla_breach', 'sum'),
                avg_days=('days_open', 'mean')
            ).reset_index()
            branch_sla['breach_rate'] = (branch_sla['breaches'] / branch_sla['total_cases'] * 100).round(1)
            branch_sla['avg_days'] = branch_sla['avg_days'].round(1)
            
            branch_sla = branch_sla.merge(
                branches_df[['branch_id', 'branch_name', 'city']], on='branch_id', how='left'
            )
            branch_sla = branch_sla.sort_values('breach_rate', ascending=False)
            
            st.dataframe(
                branch_sla[['branch_name', 'city', 'total_cases', 'breaches', 'breach_rate', 'avg_days']],
                width='stretch', hide_index=True,
                column_config={
                    'branch_name': 'Branch',
                    'city': 'City',
                    'total_cases': 'Total Cases',
                    'breaches': 'SLA Breaches',
                    'breach_rate': 'Breach Rate (%)',
                    'avg_days': 'Avg Days Open'
                }
            )
        
        # Recurrence analysis
        st.subheader("Case Recurrence Analysis")
        if len(events_df) > 0:
            recurrence = events_df.groupby('employee_id').size().reset_index(name='event_count')
            recurrence_rate = (recurrence['event_count'] > 1).mean() * 100
            repeat_offenders = recurrence[recurrence['event_count'] > 1].sort_values('event_count', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                render_kpi_card("Recurrence Rate", recurrence_rate, 'float1',
                              rating='Critical' if recurrence_rate > 12 else 'Poor' if recurrence_rate > 8 else 'Good',
                              icon='🔁')
            with col2:
                render_kpi_card("Repeat Employees", len(repeat_offenders), 'int', icon='👤')
            
            if len(repeat_offenders) > 0:
                with st.expander(f"Top {min(10, len(repeat_offenders))} Repeat Cases"):
                    # Get employee details
                    repeat_details = repeat_offenders.merge(
                        employees_df[['employee_id', 'branch_id', 'department', 'grade']],
                        on='employee_id', how='left'
                    )
                    st.dataframe(repeat_details.head(10), width='stretch', hide_index=True)
    
    # ---- TAB 3: INVESTIGATORS ----
    with tab3:
        st.subheader("Investigator Workload")
        fig = create_investigator_workload(cases_df)
        st.plotly_chart(fig, width='stretch')
        
        # Detailed investigator table
        open_cases_df = cases_df[cases_df['status'].isin(['Open', 'In Progress', 'Under Review'])]
        if len(open_cases_df) > 0:
            inv_detail = open_cases_df.groupby('investigator_id').agg(
                open_cases=('case_id', 'count'),
                avg_days_open=('days_open', 'mean'),
                categories=('category', lambda x: ', '.join(x.unique())),
                branches=('branch_id', lambda x: ', '.join(x.unique()[:5])),
                total_estimated_cost=('estimated_cost', 'sum')
            ).reset_index().sort_values('open_cases', ascending=False)
            
            inv_detail['avg_days_open'] = inv_detail['avg_days_open'].round(1)
            inv_detail['total_estimated_cost'] = inv_detail['total_estimated_cost'].round(0)
            
            st.dataframe(
                inv_detail,
                width='stretch', hide_index=True,
                column_config={
                    'investigator_id': 'Investigator',
                    'open_cases': 'Open Cases',
                    'avg_days_open': 'Avg Days Open',
                    'categories': 'Categories',
                    'branches': 'Branches',
                    'total_estimated_cost': 'Est. Cost'
                }
            )
    
    # ---- TAB 4: CASE EXPLORER ----
    with tab4:
        st.subheader("Case Explorer & Search")
        
        # Search/filter
        col1, col2, col3 = st.columns(3)
        with col1:
            search_text = st.text_input("🔍 Search Case ID / Employee", key="case_search")
        with col2:
            status_filter = st.multiselect(
                "Status", cases_df['status'].unique().tolist(),
                default=cases_df['status'].unique().tolist(),
                key="case_status_filter"
            )
        with col3:
            category_filter = st.multiselect(
                "Category", cases_df['category'].unique().tolist(),
                default=cases_df['category'].unique().tolist(),
                key="case_category_filter"
            )
        
        # Apply filters
        display_cases = cases_df.copy()
        if search_text:
            mask = display_cases['case_id'].astype(str).str.contains(search_text, case=False, na=False)
            for extra_col in ['employee_id', 'event_id']:
                if extra_col in display_cases.columns:
                    mask = mask | display_cases[extra_col].astype(str).str.contains(search_text, case=False, na=False)
            display_cases = display_cases[mask]
        if status_filter:
            display_cases = display_cases[display_cases['status'].isin(status_filter)]
        if category_filter:
            display_cases = display_cases[display_cases['category'].isin(category_filter)]
        
        # Sort by days open descending
        display_cases = display_cases.sort_values('days_open', ascending=False)
        
        # Display
        st.dataframe(
            display_cases[[
                'case_id', 'event_id', 'branch_id', 'category', 'severity',
                'status', 'stage', 'investigator_id', 'days_open', 'sla_breach',
                'estimated_cost', 'actual_cost', 'root_cause_identified', 'corrective_action_taken'
            ]].head(100),
            width='stretch', hide_index=True,
            column_config={
                'case_id': 'Case ID',
                'event_id': 'Event ID',
                'branch_id': 'Branch',
                'category': 'Category',
                'severity': 'Severity',
                'status': 'Status',
                'stage': 'Stage',
                'investigator_id': 'Investigator',
                'days_open': 'Days Open',
                'sla_breach': 'SLA Breach',
                'estimated_cost': st.column_config.NumberColumn('Est. Cost', format='₹%.0f'),
                'actual_cost': st.column_config.NumberColumn('Actual Cost', format='₹%.0f'),
                'root_cause_identified': 'Root Cause ID',
                'corrective_action_taken': 'Action Taken'
            }
        )
        
        st.caption(f"Showing {len(display_cases)} of {len(cases_df)} cases")
    
    # ============================================================
    # AI SUGGESTIONS
    # ============================================================
    st.markdown("---")
    suggestions = get_contextual_suggestions(
        insight_generator.build_ai_context(), category="case management"
    )
    render_suggestion_chips(suggestions, "cases", 
        lambda q: setattr(st.session_state, 'cases_pending_question', q))


if __name__ == "__main__":
    render()
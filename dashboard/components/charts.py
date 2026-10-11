"""
Plotly Chart Factory for ER Risk Dashboard
All chart creation functions with consistent styling
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.utils import apply_dashboard_theme, create_empty_figure, CATEGORY_COLORS, RISK_COLORS, REGION_COLORS


# ============================================================
# EXECUTIVE OVERVIEW CHARTS
# ============================================================
def create_risk_heatmap(branches_df: pd.DataFrame, 
                       kpis_df: pd.DataFrame,
                       risk_engine=None) -> go.Figure:
    """Create a Region x Category average risk heatmap (0-10 scale)"""
    if len(branches_df) == 0 or len(kpis_df) == 0:
        return create_empty_figure("Branch Risk Heatmap")
    
    # Get latest KPIs per branch
    latest_kpis = kpis_df.groupby('branch_id').last().reset_index()
    
    # Merge branch info
    merged = latest_kpis.merge(
        branches_df[['branch_id', 'branch_name', 'city', 'region']], 
        on='branch_id', how='left'
    )
    merged['region'] = merged['region'].fillna('Unknown')
    
    # Normalize each category to a 0-10 risk scale
    component_map = {
        'Grievance': ('grievance_rate', 2.0),
        'Disciplinary': ('disciplinary_rate', 3.0),
        'Absenteeism': ('absenteeism_rate', 0.5),
        'Attrition': ('attrition_rate_annualized', 0.8),
        'Whistleblower': ('whistleblower_rate', 20.0),
        'Escalation': ('grievance_escalation_rate', 1.0)
    }
    
    for col, _ in component_map.values():
        if col not in merged.columns:
            merged[col] = 0
    
    for cat, (col, factor) in component_map.items():
        merged[cat] = (
            pd.to_numeric(merged[col], errors='coerce').fillna(0) * factor
        ).clip(0, 10)
    
    cats = list(component_map.keys())
    matrix = (
        merged.groupby('region')[cats].mean()
        .reindex(sorted(merged['region'].unique()))
        .fillna(0)
    )
    
    text = [[f"{v:.1f}" for v in row] for row in matrix.values]
    
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=cats,
        y=matrix.index.tolist(),
        colorscale=[
            [0.0, '#198754'], [0.25, '#20c997'], [0.5, '#ffc107'],
            [0.75, '#fd7e14'], [1.0, '#dc3545']
        ],
        zmin=0, zmax=10,
        text=text,
        texttemplate="%{text}",
        hovertemplate="Region: %{y}<br>Category: %{x}<br>Avg Risk: %{z:.2f}/10<extra></extra>",
        colorbar=dict(title="Avg Risk")
    ))
    fig.update_layout(title="Branch Risk Heatmap: Region x Category (avg, 0-10)")
    
    return apply_dashboard_theme(fig, height=500)


def create_risk_trend_chart(kpis_df: pd.DataFrame, 
                           branches_df: pd.DataFrame = None,
                           selected_branches: List[str] = None) -> go.Figure:
    """Create risk score trend over time"""
    if len(kpis_df) == 0:
        return create_empty_figure("Risk Score Trend")
    
    # Compute risk scores if not present
    df = kpis_df.copy()
    if 'risk_score' not in df.columns:
        risk_cols = ['grievance_rate', 'disciplinary_rate', 'absenteeism_rate',
                    'attrition_rate_annualized', 'whistleblower_rate',
                    'grievance_escalation_rate']
        for c in risk_cols:
            if c not in df.columns:
                df[c] = 0
        df['risk_score'] = (
            df['grievance_rate'] * 2 * 0.25 +
            df['disciplinary_rate'] * 3 * 0.20 +
            df['absenteeism_rate'] * 0.5 * 0.15 +
            df['attrition_rate_annualized'] * 0.8 * 0.20 +
            df['whistleblower_rate'] * 20 * 0.10 +
            df['grievance_escalation_rate'] * 0.10
        ).clip(0, 10)
    
    # Aggregate by month (org level) or by branch
    if selected_branches:
        df = df[df['branch_id'].isin(selected_branches)]
        agg_df = df.groupby(['month', 'branch_id'])['risk_score'].mean().reset_index()
        
        if branches_df is not None:
            agg_df = agg_df.merge(branches_df[['branch_id', 'branch_name']], on='branch_id')
        
        fig = px.line(
            agg_df, x='month', y='risk_score', color='branch_name',
            title="Risk Score Trend by Branch",
            labels={'risk_score': 'Risk Score', 'month': 'Month', 'branch_name': 'Branch'},
            markers=True
        )
    else:
        # Org level
        agg_df = df.groupby('month')['risk_score'].mean().reset_index()
        fig = px.line(
            agg_df, x='month', y='risk_score',
            title="Organization Risk Score Trend",
            labels={'risk_score': 'Risk Score', 'month': 'Month'},
            markers=True
        )
        fig.update_traces(line_color='#1f4e79', line_width=3)
    
    # Add threshold lines
    fig.add_hline(y=3, line_dash="dash", line_color="#198754", 
                  annotation_text="Low", annotation_position="right")
    fig.add_hline(y=5, line_dash="dash", line_color="#ffc107",
                  annotation_text="Medium", annotation_position="right")
    fig.add_hline(y=7, line_dash="dash", line_color="#fd7e14",
                  annotation_text="High", annotation_position="right")
    
    return apply_dashboard_theme(fig)


def create_category_mix_chart(events_df: pd.DataFrame) -> go.Figure:
    """Create ER category distribution"""
    if len(events_df) == 0:
        return create_empty_figure("ER Category Distribution")
    
    cat_counts = events_df['category'].value_counts().reset_index()
    cat_counts.columns = ['category', 'count']
    cat_counts['pct'] = cat_counts['count'] / cat_counts['count'].sum() * 100
    
    fig = px.pie(
        cat_counts, values='count', names='category',
        title="ER Events by Category",
        color='category',
        color_discrete_map=CATEGORY_COLORS,
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
    )
    
    return apply_dashboard_theme(fig, height=400)


def create_grievance_attrition_correlation(kpis_df: pd.DataFrame) -> go.Figure:
    """Create scatter plot: Grievance rate vs Attrition rate"""
    if len(kpis_df) == 0:
        return create_empty_figure("Grievance vs Attrition Correlation")
    
    latest = kpis_df.groupby('branch_id').last().reset_index()
    
    fig = px.scatter(
        latest,
        x='grievance_rate',
        y='attrition_rate_annualized',
        size='headcount' if 'headcount' in latest.columns else None,
        color='region' if 'region' in latest.columns else None,
        color_discrete_map=REGION_COLORS,
        hover_data=['branch_name'] if 'branch_name' in latest.columns else None,
        title="Grievance Rate vs Attrition Rate by Branch",
        labels={
            'grievance_rate': 'Grievance Rate (per 100 EE)',
            'attrition_rate_annualized': 'Attrition Rate Annualized (%)'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=latest['attrition_rate_annualized'].median(), line_dash="dot", line_color="gray")
    fig.add_vline(x=latest['grievance_rate'].median(), line_dash="dot", line_color="gray")
    
    return apply_dashboard_theme(fig)


# ============================================================
# BRANCH DEEP-DIVE CHARTS
# ============================================================
def create_branch_risk_radar(branch_kpis: pd.DataFrame, 
                            branch_id: str,
                            benchmarks: Dict = None) -> go.Figure:
    """Create radar chart for branch risk profile"""
    if len(branch_kpis) == 0:
        return create_empty_figure("Branch Risk Profile")
    
    latest = branch_kpis[branch_kpis['branch_id'] == branch_id].iloc[-1]
    
    categories = ['Grievance', 'Disciplinary', 'Absenteeism', 'Attrition', 'Whistleblower', 'Escalation']
    
    # Normalize to 0-10 scale
    values = [
        min(latest.get('grievance_rate', 0) * 2, 10),
        min(latest.get('disciplinary_rate', 0) * 3, 10),
        min(latest.get('absenteeism_rate', 0) * 0.5, 10),
        min(latest.get('attrition_rate_annualized', 0) * 0.8, 10),
        min(latest.get('whistleblower_rate', 0) * 20, 10),
        min(latest.get('grievance_escalation_rate', 0), 10)
    ]
    
    # Benchmark values (industry average)
    benchmark_values = [4, 2, 5, 10, 0.1, 15]  # Rough industry averages normalized
    benchmark_values = [min(v, 10) for v in benchmark_values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Branch',
        line_color='#1f4e79',
        fillcolor='rgba(31, 78, 121, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values + [benchmark_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Industry Avg',
        line_color='#6c757d',
        fillcolor='rgba(108, 117, 125, 0.1)',
        line_dash='dash'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10]),
            angularaxis=dict(direction="clockwise")
        ),
        title=f"Risk Profile: {latest.get('branch_name', branch_id)}",
        showlegend=True
    )
    
    return apply_dashboard_theme(fig, height=450)


def create_department_breakdown(employees_df: pd.DataFrame, 
                               events_df: pd.DataFrame,
                               branch_id: str) -> go.Figure:
    """Create department-level risk breakdown"""
    if len(employees_df) == 0:
        return create_empty_figure("Department Risk Breakdown")
    
    branch_emp = employees_df[employees_df['branch_id'] == branch_id]
    branch_events = events_df[events_df['branch_id'] == branch_id]
    
    if len(branch_emp) == 0:
        return create_empty_figure("Department Risk Breakdown")
    
    # Aggregate by department
    dept_stats = branch_emp.groupby('department').agg(
        headcount=('employee_id', 'count'),
        avg_performance=('performance_rating', 'mean'),
        avg_tenure=('tenure_years', 'mean')
    ).reset_index()
    
    # Add event rates
    if len(branch_events) > 0:
        # Need to map employee to department - merge via employee_id
        emp_dept = branch_emp[['employee_id', 'department']]
        events_with_dept = branch_events.merge(emp_dept, on='employee_id', how='left')
        dept_events = events_with_dept.groupby('department').size().reset_index(name='event_count')
        
        dept_stats = dept_stats.merge(dept_events, on='department', how='left')
        dept_stats['event_count'] = dept_stats['event_count'].fillna(0)
        dept_stats['event_rate'] = dept_stats['event_count'] / dept_stats['headcount'] * 100
    else:
        dept_stats['event_count'] = 0
        dept_stats['event_rate'] = 0
    
    dept_stats = dept_stats.sort_values('event_rate', ascending=True)
    
    fig = px.bar(
        dept_stats, x='event_rate', y='department', orientation='h',
        color='avg_performance',
        color_continuous_scale='RdYlGn',
        title="ER Event Rate by Department",
        labels={'event_rate': 'Events per 100 EE', 'department': 'Department', 'avg_performance': 'Avg Performance'},
        text='event_rate'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    
    return apply_dashboard_theme(fig, height=400)


def create_tenure_risk_matrix(employees_df: pd.DataFrame, 
                             events_df: pd.DataFrame,
                             branch_id: str) -> go.Figure:
    """Create tenure vs risk scatter"""
    if len(employees_df) == 0:
        return create_empty_figure("Tenure vs Risk Matrix")
    
    branch_emp = employees_df[employees_df['branch_id'] == branch_id].copy()
    
    if len(branch_emp) == 0:
        return create_empty_figure("Tenure vs Risk Matrix")
    
    # Add event counts
    branch_events = events_df[events_df['branch_id'] == branch_id]
    if len(branch_events) > 0:
        emp_events = branch_events.groupby('employee_id').agg(
            event_count=('event_id', 'count'),
            max_severity=('severity', 'max'),
            categories=('category', lambda x: ', '.join(x.unique()))
        ).reset_index()
        
        branch_emp = branch_emp.merge(emp_events, on='employee_id', how='left')
        branch_emp['event_count'] = branch_emp['event_count'].fillna(0)
        branch_emp['max_severity'] = branch_emp['max_severity'].fillna(0)
    else:
        branch_emp['event_count'] = 0
        branch_emp['max_severity'] = 0
        branch_emp['categories'] = ''
    
    # Risk score
    branch_emp['risk_score'] = (
        branch_emp['event_count'] * 0.5 +
        branch_emp['max_severity'] * 1.5 +
        (5 - branch_emp['performance_rating']) * 1.0
    ).clip(0, 10)
    
    fig = px.scatter(
        branch_emp,
        x='tenure_years',
        y='risk_score',
        color='department',
        size='event_count',
        hover_data=['employee_id', 'grade', 'performance_rating', 'categories'],
        title="Employee Tenure vs Risk Score",
        labels={'tenure_years': 'Tenure (Years)', 'risk_score': 'Risk Score'}
    )
    
    # Add quadrant lines
    fig.add_hline(y=5, line_dash="dot", line_color="gray")
    fig.add_vline(x=branch_emp['tenure_years'].median(), line_dash="dot", line_color="gray")
    
    # Annotate quadrants
    fig.add_annotation(x=branch_emp['tenure_years'].max()*0.8, y=8,
                      text="High Risk, High Tenure", showarrow=False, font=dict(size=10, color="red"))
    fig.add_annotation(x=branch_emp['tenure_years'].max()*0.2, y=8,
                      text="High Risk, Low Tenure", showarrow=False, font=dict(size=10, color="orange"))
    
    return apply_dashboard_theme(fig, height=450)


def create_peer_benchmarking(kpis_df: pd.DataFrame, 
                            branches_df: pd.DataFrame,
                            branch_id: str,
                            metric: str = 'risk_score') -> go.Figure:
    """Create peer benchmarking chart"""
    if len(kpis_df) == 0:
        return create_empty_figure("Peer Benchmarking")
    
    latest = kpis_df.groupby('branch_id').last().reset_index()
    
    if 'risk_score' not in latest.columns:
        latest['risk_score'] = (
            latest['grievance_rate'] * 2 * 0.25 +
            latest['disciplinary_rate'] * 3 * 0.20 +
            latest['absenteeism_rate'] * 0.5 * 0.15 +
            latest['attrition_rate_annualized'] * 0.8 * 0.20 +
            latest['whistleblower_rate'] * 20 * 0.10
        ).clip(0, 10)
    
    latest = latest.merge(branches_df[['branch_id', 'branch_name', 'city', 'region']], on='branch_id')
    
    # Get region of selected branch
    branch_region = latest[latest['branch_id'] == branch_id]['region'].values
    branch_region = branch_region[0] if len(branch_region) > 0 else None
    
    # Filter to same region for peer comparison
    if branch_region:
        peers = latest[latest['region'] == branch_region].copy()
    else:
        peers = latest.copy()
    
    peers = peers.sort_values(metric, ascending=True)
    
    # Highlight selected branch
    colors = ['#dc3545' if b == branch_id else '#1f4e79' for b in peers['branch_id']]
    
    fig = go.Figure(go.Bar(
        x=peers[metric],
        y=peers['branch_name'] + ' (' + peers['city'] + ')',
        orientation='h',
        marker_color=colors,
        text=peers[metric].round(2),
        textposition='outside'
    ))
    
    # Add average line
    avg_val = peers[metric].mean()
    fig.add_vline(x=avg_val, line_dash="dash", line_color="gray",
                  annotation_text=f"Region Avg: {avg_val:.2f}")
    
    fig.update_layout(
        title=f"Peer Benchmarking: {metric.replace('_', ' ').title()} ({branch_region or 'All Regions'})",
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title="Branch"
    )
    
    return apply_dashboard_theme(fig, height=450)


# ============================================================
# PREDICTIVE ANALYTICS CHARTS
# ============================================================
def create_attrition_risk_distribution(attrition_predictions: pd.DataFrame) -> go.Figure:
    """Create histogram of attrition risk scores"""
    if len(attrition_predictions) == 0:
        return create_empty_figure("Attrition Risk Distribution")
    
    risk_col = 'attrition_risk' if 'attrition_risk' in attrition_predictions.columns else 'risk_score'
    
    fig = px.histogram(
        attrition_predictions,
        x=risk_col,
        nbins=20,
        title="Employee Attrition Risk Distribution",
        labels={risk_col: 'Attrition Risk Probability', 'count': 'Number of Employees'},
        color_discrete_sequence=['#1f4e79']
    )
    
    # Add threshold lines
    fig.add_vline(x=0.5, line_dash="dash", line_color="#ffc107", annotation_text="Medium Risk")
    fig.add_vline(x=0.7, line_dash="dash", line_color="#fd7e14", annotation_text="High Risk")
    fig.add_vline(x=0.85, line_dash="dash", line_color="#dc3545", annotation_text="Critical Risk")
    
    return apply_dashboard_theme(fig)


def create_escalation_risk_scatter(escalation_predictions: pd.DataFrame) -> go.Figure:
    """Create scatter of grievance escalation risk"""
    if len(escalation_predictions) == 0:
        return create_empty_figure("Grievance Escalation Risk")
    
    fig = px.scatter(
        escalation_predictions,
        x='severity',
        y='escalation_risk',
        color='branch_id' if 'branch_id' in escalation_predictions.columns else None,
        size='escalation_risk',
        hover_data=['event_id', 'employee_id', 'subtype'],
        title="Grievance Escalation Risk by Severity",
        labels={'severity': 'Severity (1-5)', 'escalation_risk': 'Escalation Probability'}
    )
    
    # Add risk zones
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="red", opacity=0.1, annotation_text="Critical")
    fig.add_hrect(y0=0.5, y1=0.7, fillcolor="orange", opacity=0.1, annotation_text="High")
    fig.add_hrect(y0=0.3, y1=0.5, fillcolor="yellow", opacity=0.1, annotation_text="Medium")
    
    return apply_dashboard_theme(fig)


def create_hotspot_map(hotspot_data: pd.DataFrame) -> go.Figure:
    """Create hotspot visualization (bar chart version for simplicity)"""
    if len(hotspot_data) == 0:
        return create_empty_figure("Risk Hotspots")
    
    # Create a more detailed hotspot view
    fig = px.bar(
        hotspot_data,
        x='avg_risk_score',
        y='top_branches',
        orientation='h',
        color='avg_risk_score',
        color_continuous_scale=['#198754', '#ffc107', '#fd7e14', '#dc3545'],
        title="Risk Hotspot Clusters",
        labels={'avg_risk_score': 'Avg Cluster Risk Score', 'top_branches': 'Branches in Cluster'}
    )
    
    return apply_dashboard_theme(fig, height=400)


def create_forecast_chart(historical_df: pd.DataFrame,
                         forecast_df: pd.DataFrame,
                         metric_name: str,
                         title: str) -> go.Figure:
    """Create forecast chart with historical data and predictions"""
    if historical_df is None or len(historical_df) == 0 or metric_name not in historical_df.columns:
        return create_empty_figure(title)
    
    fig = go.Figure()
    
    # Historical (stringify x so it joins cleanly with YYYY-MM forecast labels)
    fig.add_trace(go.Scatter(
        x=historical_df['month'].astype(str),
        y=historical_df[metric_name],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f4e79', width=3),
        marker=dict(size=6)
    ))
    
    if forecast_df is not None and len(forecast_df) > 0:
        fc_months = forecast_df['month'].astype(str).tolist()
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=fc_months,
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#fd7e14', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=fc_months + fc_months[::-1],
            y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(253, 126, 20, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% Confidence',
            showlegend=True
        ))
    
    fig.update_layout(title=title)
    
    return apply_dashboard_theme(fig)


def create_what_if_chart(simulation_results: List[Dict]) -> go.Figure:
    """Create what-if simulation comparison"""
    if not simulation_results:
        return create_empty_figure("What-If Simulation")
    
    df = pd.DataFrame(simulation_results)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['intervention'],
        y=df['current_risk'],
        name='Current Risk',
        marker_color='#6c757d',
        text=df['current_risk'],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=df['intervention'],
        y=df['projected_risk'],
        name='Projected Risk',
        marker_color='#1f4e79',
        text=df['projected_risk'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="What-If: Intervention Impact on Risk Score",
        xaxis_title="Intervention",
        yaxis_title="Risk Score",
        barmode='group'
    )
    
    return apply_dashboard_theme(fig)


# ============================================================
# CASE MANAGEMENT CHARTS
# ============================================================
def create_case_funnel(cases_df: pd.DataFrame) -> go.Figure:
    """Create case lifecycle funnel"""
    if len(cases_df) == 0:
        return create_empty_figure("Case Lifecycle Funnel")
    
    stages = ['Open', 'Intake', 'Investigation', 'Review', 'Decision', 'Resolved', 'Closed']
    stage_counts = []
    
    for stage in stages:
        count = len(cases_df[cases_df['stage'] == stage]) if 'stage' in cases_df.columns else 0
        stage_counts.append(count)
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=stage_counts,
        textinfo="value+percent initial",
        marker=dict(color=['#dc3545', '#fd7e14', '#ffc107', '#20c997', '#198754', '#1f4e79', '#6c757d'])
    ))
    
    fig.update_layout(title="Case Lifecycle Funnel")
    
    return apply_dashboard_theme(fig, height=450)


def create_sla_breach_chart(cases_df: pd.DataFrame) -> go.Figure:
    """Create SLA breach rate by category"""
    if len(cases_df) == 0:
        return create_empty_figure("SLA Breach Rate by Category")
    
    sla_by_cat = cases_df.groupby('category').agg(
        total=('case_id', 'count'),
        breaches=('sla_breach', 'sum')
    ).reset_index()
    sla_by_cat['breach_rate'] = sla_by_cat['breaches'] / sla_by_cat['total'] * 100
    
    fig = px.bar(
        sla_by_cat, x='category', y='breach_rate',
        color='category', color_discrete_map=CATEGORY_COLORS,
        title="SLA Breach Rate by Category",
        labels={'breach_rate': 'SLA Breach Rate (%)', 'category': 'Category'},
        text='breach_rate'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    return apply_dashboard_theme(fig)


def create_resolution_time_box(cases_df: pd.DataFrame) -> go.Figure:
    """Create box plot of resolution times by category"""
    if len(cases_df) == 0:
        return create_empty_figure("Resolution Time Distribution")
    
    resolved = cases_df[cases_df['status'].isin(['Resolved', 'Closed'])]
    
    if len(resolved) == 0:
        return create_empty_figure("Resolution Time Distribution")
    
    fig = px.box(
        resolved, x='category', y='days_open',
        color='category', color_discrete_map=CATEGORY_COLORS,
        title="Resolution Time Distribution by Category",
        labels={'days_open': 'Resolution Time (Days)', 'category': 'Category'},
        points='outliers'
    )
    
    return apply_dashboard_theme(fig, height=400)


def create_investigator_workload(cases_df: pd.DataFrame) -> go.Figure:
    """Create investigator workload chart"""
    if len(cases_df) == 0:
        return create_empty_figure("Investigator Workload")
    
    open_cases = cases_df[cases_df['status'].isin(['Open', 'In Progress', 'Under Review'])]
    
    if len(open_cases) == 0:
        return create_empty_figure("Investigator Workload")
    
    workload = open_cases.groupby('investigator_id').agg(
        open_cases=('case_id', 'count'),
        avg_days=('days_open', 'mean'),
        categories=('category', lambda x: ', '.join(x.unique()[:3]))
    ).reset_index().sort_values('open_cases', ascending=True)
    
    fig = px.bar(
        workload, x='open_cases', y='investigator_id', orientation='h',
        color='avg_days',
        color_continuous_scale='RdYlGn_r',
        title="Investigator Workload (Open Cases)",
        labels={'open_cases': 'Open Cases', 'investigator_id': 'Investigator', 'avg_days': 'Avg Days Open'},
        text='open_cases',
        hover_data=['categories']
    )
    
    fig.update_traces(textposition='outside')
    
    return apply_dashboard_theme(fig, height=400)


# ============================================================
# ADMIN / MONITORING CHARTS
# ============================================================
def create_model_performance_chart(model_metrics: Dict) -> go.Figure:
    """Create model performance monitoring"""
    if not model_metrics:
        return create_empty_figure("Model Performance")
    
    metrics = []
    for model_name, metric in model_metrics.items():
        if isinstance(metric, dict):
            metrics.append({
                'model': model_name,
                'auc': metric.get('auc', 0),
                'n_samples': metric.get('n_samples', 0)
            })
    
    if not metrics:
        return create_empty_figure("Model Performance")
    
    df = pd.DataFrame(metrics)
    
    fig = px.bar(
        df, x='model', y='auc',
        color='auc', color_continuous_scale='RdYlGn',
        title="Model AUC Scores",
        text='auc',
        hover_data=['n_samples']
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.add_hline(y=0.7, line_dash="dash", line_color="gray", annotation_text="Acceptable")
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Good")
    
    return apply_dashboard_theme(fig)


def create_data_quality_chart(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create data quality overview"""
    tables = ['branches', 'employees', 'events', 'cases', 'kpis']
    quality_data = []
    
    for table in tables:
        df = data_dict.get(table, pd.DataFrame())
        quality_data.append({
            'table': table,
            'records': len(df),
            'columns': len(df.columns) if len(df) > 0 else 0,
            'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 100
        })
    
    df = pd.DataFrame(quality_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['table'],
        y=df['records'],
        name='Records',
        marker_color='#1f4e79',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['table'],
        y=df['missing_pct'],
        name='Missing %',
        marker_color='#dc3545',
        yaxis='y2',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Data Quality Overview",
        yaxis=dict(title="Record Count", side="left"),
        yaxis2=dict(title="Missing %", side="right", overlaying="y", range=[0, 100])
    )
    
    return apply_dashboard_theme(fig)
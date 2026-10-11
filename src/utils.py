"""
Utility Functions for ER Risk Dashboard
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Any, Optional
import hashlib
import json


# ============================================================
# DATA LOADING & CACHING
# ============================================================
@st.cache_data(show_spinner="Loading ER data...")
def load_all_data(data_path: str = "data") -> Dict[str, pd.DataFrame]:
    """Load all parquet data files"""
    data = {}
    
    files = {
        'branches': 'branches.parquet',
        'employees': 'employees.parquet',
        'events': 'events.parquet',
        'cases': 'cases.parquet',
        'kpis': 'monthly_kpis.parquet'
    }
    
    for key, filename in files.items():
        filepath = f"{data_path}/{filename}"
        try:
            data[key] = pd.read_parquet(filepath)
            # Ensure datetime columns
            if key in ['events', 'cases']:
                date_cols = [c for c in data[key].columns if 'date' in c.lower()]
                for col in date_cols:
                    data[key][col] = pd.to_datetime(data[key][col], errors='coerce')
        except FileNotFoundError:
            st.warning(f"Data file not found: {filepath}")
            data[key] = pd.DataFrame()
    
    return data


@st.cache_resource
def initialize_risk_engine(data_dict: Dict[str, pd.DataFrame]):
    """Initialize and cache the risk engine"""
    from src.risk_engine import create_risk_engine
    from src.insight_generator import InsightGenerator
    import src.kpi_calculator as kpi_module

    engine = create_risk_engine(
        data_dict['events'],
        data_dict['kpis'],
        data_dict['branches'],
        data_dict['employees'],
        data_dict.get('cases')
    )

    insight_gen = InsightGenerator(engine, kpi_module)

    return engine, kpi_module, insight_gen


# ============================================================
# FILTER HELPERS
# ============================================================
def get_filter_options(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List]:
    """Extract filter options from data"""
    branches_df = data_dict['branches']
    events_df = data_dict['events']
    employees_df = data_dict['employees']
    
    options = {
        'regions': sorted(branches_df['region'].unique().tolist()) if 'region' in branches_df.columns else [],
        'cities': sorted(branches_df['city'].unique().tolist()) if 'city' in branches_df.columns else [],
        'branches': sorted(branches_df['branch_name'].unique().tolist()) if 'branch_name' in branches_df.columns else [],
        'branch_ids': sorted(branches_df['branch_id'].unique().tolist()) if 'branch_id' in branches_df.columns else [],
        'categories': sorted(events_df['category'].unique().tolist()) if 'category' in events_df.columns else [],
        'subtypes': sorted(events_df['subtype'].unique().tolist()) if 'subtype' in events_df.columns else [],
        'departments': sorted(employees_df['department'].unique().tolist()) if 'department' in employees_df.columns else [],
        'grades': sorted(employees_df['grade'].unique().tolist()) if 'grade' in employees_df.columns else [],
        'months': sorted(events_df['month'].unique().tolist()) if 'month' in events_df.columns else [],
        'years': sorted(events_df['year'].unique().tolist()) if 'year' in events_df.columns else []
    }
    
    return options


def apply_filters(events_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to events dataframe"""
    df = events_df.copy()
    
    if filters.get('branch_ids'):
        df = df[df['branch_id'].isin(filters['branch_ids'])]
    
    if filters.get('categories'):
        df = df[df['category'].isin(filters['categories'])]
    
    if filters.get('subtypes'):
        df = df[df['subtype'].isin(filters['subtypes'])]
    
    if filters.get('date_range'):
        start, end = filters['date_range']
        df = df[(df['event_date'] >= pd.Timestamp(start)) & (df['event_date'] <= pd.Timestamp(end))]
    
    if filters.get('severity_range'):
        min_sev, max_sev = filters['severity_range']
        df = df[(df['severity'] >= min_sev) & (df['severity'] <= max_sev)]
    
    if filters.get('escalated_only'):
        df = df[df['escalated'] == True]
    
    if filters.get('resolved_only'):
        df = df[df['resolved'] == True]
    
    return df


# ============================================================
# FORMATTING HELPERS
# ============================================================
def format_number(value: float, fmt: str = 'auto') -> str:
    """Format numbers for display"""
    if pd.isna(value):
        return "—"
    
    if fmt == 'pct':
        return f"{value:.1f}%"
    elif fmt == 'pct2':
        return f"{value:.2f}%"
    elif fmt == 'int':
        return f"{int(value):,}"
    elif fmt == 'float1':
        return f"{value:.1f}"
    elif fmt == 'float2':
        return f"{value:.2f}"
    elif fmt == 'currency':
        return f"₹{value:,.0f}"
    elif fmt == 'auto':
        if value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}K"
        elif value >= 100:
            return f"{value:.0f}"
        elif value >= 1:
            return f"{value:.1f}"
        else:
            return f"{value:.2f}"
    return str(value)


def get_risk_color(score: float) -> str:
    """Get color for risk score"""
    if score >= 8.5:
        return '#dc3545'  # Critical - Red
    elif score >= 7:
        return '#fd7e14'  # High - Orange
    elif score >= 5:
        return '#ffc107'  # Medium - Yellow/Amber
    elif score >= 3:
        return '#20c997'  # Low-Medium - Teal
    else:
        return '#198754'  # Low - Green


def get_risk_level(score: float) -> str:
    """Get risk level label"""
    if score >= 8.5:
        return 'Critical'
    elif score >= 7:
        return 'High'
    elif score >= 5:
        return 'Medium'
    elif score >= 3:
        return 'Low-Medium'
    else:
        return 'Low'


def get_trend_icon(trend: str) -> str:
    """Get icon for trend direction"""
    icons = {
        'Worsening': '',
        'Improving': '',
        'Stable': '',
        'Up': '',
        'Down': '',
        'Flat': ''
    }
    return icons.get(trend, '')


def get_trend_color(trend: str) -> str:
    """Get color for trend"""
    colors = {
        'Worsening': '#dc3545',
        'Improving': '#198754',
        'Stable': '#6c757d',
        'Up': '#dc3545',
        'Down': '#198754',
        'Flat': '#6c757d'
    }
    return colors.get(trend, '#6c757d')


# ============================================================
# CHART HELPERS
# ============================================================
def create_empty_figure(title: str, height: int = 300):
    """Create empty figure with message"""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text="No data available for selected filters",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#6c757d")
    )
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def apply_dashboard_theme(fig, height: int = 400):
    """Apply consistent theme to plotly figures"""
    fig.update_layout(
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color="#1a1a2e"),
        title=dict(font=dict(size=16, color="#1f4e79"), x=0.02, xanchor='left'),
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e0e0e0',
            borderwidth=1
        ),
        hovermode='x unified',
        xaxis=dict(
            showgrid=True, gridcolor='#f0f0f0',
            linecolor='#e0e0e0', linewidth=1,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#f0f0f0',
            linecolor='#e0e0e0', linewidth=1,
            tickfont=dict(size=11)
        )
    )
    return fig


# ============================================================
# SESSION STATE HELPERS
# ============================================================
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'filters': {
            'branch_ids': [],
            'categories': [],
            'subtypes': [],
            'date_range': None,
            'severity_range': (1, 5),
            'escalated_only': False,
            'resolved_only': False
        },
        'selected_branch': None,
        'chat_history': [],
        'ai_mode': 'Template (Instant)',
        'last_data_refresh': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_filters():
    """Reset all filters to defaults"""
    st.session_state.filters = {
        'branch_ids': [],
        'categories': [],
        'subtypes': [],
        'date_range': None,
        'severity_range': (1, 5),
        'escalated_only': False,
        'resolved_only': False
    }


# ============================================================
# EXPORT HELPERS
# ============================================================
def download_dataframe(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """Create download button for dataframe"""
    csv = df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv',
        width='stretch'
    )


def export_insights_to_json(insights: Dict, filename: str = "er_insights.json"):
    """Export insights as JSON"""
    json_str = json.dumps(insights, indent=2, default=str)
    st.download_button(
        label="Download Insights (JSON)",
        data=json_str,
        file_name=filename,
        mime='application/json',
        width='stretch'
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================
def validate_data_quality(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Check data quality and completeness"""
    issues = []
    warnings = []
    
    # Check required tables
    required = ['branches', 'employees', 'events', 'kpis']
    for table in required:
        if table not in data_dict or len(data_dict[table]) == 0:
            issues.append(f"Missing or empty table: {table}")
    
    if 'events' in data_dict and len(data_dict['events']) > 0:
        events = data_dict['events']
        # Check date range
        if 'event_date' in events.columns:
            date_range = events['event_date'].max() - events['event_date'].min()
            if date_range.days < 90:
                warnings.append(f"Event data spans only {date_range.days} days")
        
        # Check categories
        expected_cats = ['Grievance', 'Disciplinary', 'Absenteeism', 'Attrition', 'Whistleblower', 'Staff Accountability']
        missing_cats = [c for c in expected_cats if c not in events['category'].unique()]
        if missing_cats:
            warnings.append(f"Missing event categories: {missing_cats}")
    
    if 'branches' in data_dict:
        if len(data_dict['branches']) < 10:
            warnings.append(f"Only {len(data_dict['branches'])} branches - limited benchmarking")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'record_counts': {k: len(v) for k, v in data_dict.items() if hasattr(v, '__len__')}
    }


# ============================================================
# COLOR PALETTES
# ============================================================
CATEGORY_COLORS = {
    'Grievance': '#1f4e79',
    'Disciplinary': '#dc3545',
    'Absenteeism': '#fd7e14',
    'Attrition': '#6f42c1',
    'Whistleblower': '#20c997',
    'Staff Accountability': '#ffc107'
}

RISK_COLORS = ['#198754', '#20c997', '#ffc107', '#fd7e14', '#dc3545']
REGION_COLORS = {
    'North': '#1f4e79',
    'South': '#20c997',
    'East': '#fd7e14',
    'West': '#6f42c1',
    'Central': '#ffc107'
}

SEVERITY_COLORS = {
    1: '#198754', 2: '#20c997', 3: '#ffc107', 4: '#fd7e14', 5: '#dc3545'
}

STATUS_COLORS = {
    'Open': '#dc3545',
    'In Progress': '#fd7e14',
    'Under Review': '#ffc107',
    'Resolved': '#198754',
    'Closed': '#6c757d',
    'Escalated': '#6f42c1'
}
"""
KPI Card Components for Streamlit Dashboard
Reusable, styled metric cards with ratings and trends - using native Streamlit components
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.utils import format_number, get_risk_color, get_risk_level, get_trend_icon, get_trend_color


# Custom CSS for KPI cards - injected once
_KPI_CARD_CSS = """
<style>
.kpi-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #1f4e79;
    border-radius: 8px;
    padding: 16px;
    height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s, transform 0.2s;
}
.kpi-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.kpi-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 8px;
}
.kpi-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.kpi-title {
    color: #6c757d;
    font-size: 0.85rem;
    font-weight: 500;
    line-height: 1.3;
}
.kpi-icon {
    font-size: 1.5rem;
    line-height: 1;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
    margin: 4px 0;
    word-break: break-word;
}
.kpi-trend {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 4px;
    font-size: 0.85rem;
    font-weight: 500;
}
.kpi-help {
    color: #6c757d;
    font-size: 0.75rem;
    margin-top: 4px;
    line-height: 1.4;
}
.rating-badge {
    display: inline-block;
    background: var(--badge-bg, #1f4e7920);
    color: var(--badge-color, #1f4e79);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.trend-up { color: #dc3545; }
.trend-down { color: #198754; }
.trend-flat { color: #6c757d; }
</style>
"""


def _inject_kpi_css():
    """Inject CSS for KPI cards (safe to call multiple times)"""
    st.markdown(_KPI_CARD_CSS, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: Any,
    fmt: str = 'auto',
    rating: str = None,
    color: str = None,
    trend: str = None,
    trend_value: float = None,
    help_text: str = None,
    icon: str = None,
    key: str = None
):
    """Render a single self-contained KPI card (one HTML block)"""

    if '_kpi_css_injected' not in st.session_state:
        _inject_kpi_css()
        st.session_state._kpi_css_injected = True

    if isinstance(value, (int, float)) and not pd.isna(value):
        display_value = format_number(value, fmt)
    else:
        display_value = str(value) if value is not None else "—"

    color_map = {
        'Excellent': '#198754', 'Good': '#20c997', 'Average': '#ffc107',
        'Poor': '#fd7e14', 'Critical': '#dc3545',
        'Low': '#198754', 'Low-Medium': '#20c997', 'Medium': '#ffc107',
        'High': '#fd7e14'
    }

    if color is None:
        color = color_map.get(rating, '#1f4e79') if rating else '#1f4e79'

    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ''

    badge_html = ''
    if rating:
        badge_color = color_map.get(rating, color)
        badge_html = (
            f'<span class="rating-badge" style="--badge-bg: {badge_color}20; '
            f'--badge-color: {badge_color};">{rating}</span>'
        )

    trend_html = ''
    if trend and trend_value is not None:
        trend_class = "trend-up" if trend in ("Worsening", "Up") else "trend-down" if trend in ("Improving", "Down") else "trend-flat"
        trend_html = f'<div class="kpi-trend {trend_class}">{abs(trend_value):.1f}% vs prior</div>'

    help_html = f'<div class="kpi-help">{help_text}</div>' if help_text else ''

    card_html = (
        f'<div class="kpi-card" style="border-left-color: {color};">'
        f'<div class="kpi-header">'
        f'<div class="kpi-title-row">{icon_html}<span class="kpi-title">{label}</span></div>'
        f'{badge_html}'
        f'</div>'
        f'<div class="kpi-value">{display_value}</div>'
        f'{trend_html}{help_html}'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def render_kpi_card_simple(
    label: str,
    value: Any,
    fmt: str = 'auto',
    delta: str = None,
    delta_color: str = "normal",
    help_text: str = None,
    icon: str = None
):
    """Simplified KPI card using st.metric (for simpler cases)"""
    from src.utils import format_number
    
    if isinstance(value, (int, float)) and not pd.isna(value):
        display_value = format_number(value, fmt)
    else:
        display_value = str(value) if value is not None else "—"
    
    label_with_icon = f"{icon} {label}" if icon else label
    
    st.metric(
        label=label_with_icon,
        value=display_value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def render_kpi_row(kpis: List[Dict], columns: int = 4):
    """Render a row of KPI cards"""
    cols = st.columns(columns)
    
    for i, kpi in enumerate(kpis):
        with cols[i % columns]:
            render_kpi_card(
                label=kpi.get('label', ''),
                value=kpi.get('value', 0),
                fmt=kpi.get('fmt', 'auto'),
                rating=kpi.get('rating'),
                color=kpi.get('color'),
                trend=kpi.get('trend'),
                trend_value=kpi.get('trend_value'),
                help_text=kpi.get('help_text'),
                icon=kpi.get('icon')
            )


def render_risk_gauge(score: float, label: str = "Risk Score", size: int = 200):
    """Render a risk score gauge chart"""
    import plotly.graph_objects as go
    
    # Determine color
    if score >= 8.5:
        gauge_color = "#dc3545"
        level = "Critical"
    elif score >= 7:
        gauge_color = "#fd7e14"
        level = "High"
    elif score >= 5:
        gauge_color = "#ffc107"
        level = "Medium"
    elif score >= 3:
        gauge_color = "#20c997"
        level = "Low-Medium"
    else:
        gauge_color = "#198754"
        level = "Low"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 16, 'color': '#1f4e79'}},
        number={'font': {'size': 40, 'color': gauge_color}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#6c757d"},
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 3], 'color': '#198754'},
                {'range': [3, 5], 'color': '#20c997'},
                {'range': [5, 7], 'color': '#ffc107'},
                {'range': [7, 8.5], 'color': '#fd7e14'},
                {'range': [8.5, 10], 'color': '#dc3545'}
            ],
            'threshold': {
                'line': {'color': "#1a1a2e", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=size,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'}
    )
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def render_metric_comparison(
    current: float,
    previous: float,
    label: str,
    fmt: str = 'float1',
    higher_is_better: bool = False
):
    """Render metric with comparison to previous period"""
    delta = current - previous
    pct_change = (delta / previous * 100) if previous != 0 else 0
    
    if higher_is_better:
        is_good = delta > 0
    else:
        is_good = delta < 0
    
    trend = "Improving" if is_good else "Worsening"
    trend_color = "#198754" if is_good else "#dc3545"
    
    render_kpi_card(
        label=label,
        value=current,
        fmt=fmt,
        trend=trend,
        trend_value=abs(pct_change),
        color=trend_color
    )


def render_kpi_table(kpi_df: pd.DataFrame, 
                    show_benchmark: bool = True,
                    key: str = None):
    """Render KPI table with ratings and benchmarks"""
    if len(kpi_df) == 0:
        st.info("No KPI data available")
        return
    
    # Prepare display dataframe
    display_df = kpi_df[['kpi_name', 'value', 'rating', 'color']].copy()
    display_df.columns = ['KPI', 'Value', 'Rating', 'Color']
    
    if show_benchmark and 'benchmark_average' in kpi_df.columns:
        display_df['Benchmark (Avg)'] = kpi_df['benchmark_average'].round(1)
        display_df['Gap'] = (display_df['Value'] - display_df['Benchmark (Avg)']).round(1)
    
    # Style function
    def style_rating(val):
        row = display_df[display_df['Rating'] == val].iloc[0] if len(display_df[display_df['Rating'] == val]) > 0 else None
        if row is not None:
            return f'background-color: {row["Color"]}20; color: {row["Color"]}; font-weight: 600'
        return ''
    
    def style_gap(val):
        if pd.isna(val):
            return ''
        if val < 0:
            return 'color: #198754'
        elif val > 0:
            return 'color: #dc3545'
        return ''
    
    styled = display_df.style.map(style_rating, subset=['Rating'])
    if 'Gap' in display_df.columns:
        styled = styled.map(style_gap, subset=['Gap'])
    
    st.dataframe(
        styled.format({'Value': '{:.2f}', 'Benchmark (Avg)': '{:.1f}', 'Gap': '{:+.1f}'}),
        width='stretch',
        hide_index=True,
        key=key
    )


def render_category_kpi_cards(events_df: pd.DataFrame, 
                             employees_df: pd.DataFrame,
                             categories: List[str] = None):
    """Render KPI cards for each ER category"""
    if categories is None:
        categories = ['Grievance', 'Disciplinary', 'Absenteeism', 'Attrition', 'Whistleblower']
    
    total_emp = len(employees_df)
    kpis = []
    
    for cat in categories:
        cat_events = events_df[events_df['category'] == cat]
        count = len(cat_events)
        rate = count / total_emp * 100 if total_emp > 0 else 0
        
        # Escalation rate for grievance/disciplinary
        if cat in ['Grievance', 'Disciplinary']:
            esc_rate = cat_events['escalated'].mean() * 100 if len(cat_events) > 0 else 0
            help_text = f"{count} cases | Escalation: {esc_rate:.1f}%"
        elif cat == 'Absenteeism':
            help_text = f"{count} events"
        elif cat == 'Attrition':
            vol_pct = (cat_events['subtype'] == 'Voluntary').mean() * 100 if len(cat_events) > 0 else 0
            help_text = f"{count} leavers | {vol_pct:.0f}% voluntary"
        else:
            help_text = f"{count} reports"
        
        kpis.append({
            'label': cat,
            'value': rate,
            'fmt': 'float2',
            'help_text': help_text,
            'icon': {'Grievance': '', 'Disciplinary': '', 'Absenteeism': '', 
                    'Attrition': '', 'Whistleblower': ''}.get(cat, '')
        })
    
    render_kpi_row(kpis, columns=len(kpis))


def render_executive_kpis(insights: Dict[str, Any]):
    """Render executive-level KPI cards from insights"""
    stats = insights.get('summary_stats', {})
    
    kpis = [
        {
            'label': 'Org Risk Score',
            'value': stats.get('org_risk_score', 0),
            'fmt': 'float1',
            'rating': stats.get('risk_level', 'Medium'),
            'icon': '',
            'help_text': f"Across {stats.get('total_branches', 0)} branches"
        },
        {
            'label': 'High Risk Branches',
            'value': stats.get('high_risk_branches', 0),
            'fmt': 'int',
            'rating': 'Critical' if stats.get('high_risk_branches', 0) > 10 else 'Poor' if stats.get('high_risk_branches', 0) > 5 else 'Average',
            'icon': '',
            'help_text': "Risk score ≥ 7"
        },
        {
            'label': 'Critical Branches',
            'value': stats.get('critical_branches', 0),
            'fmt': 'int',
            'rating': 'Critical' if stats.get('critical_branches', 0) > 0 else 'Excellent',
            'icon': '',
            'help_text': "Risk score ≥ 8.5"
        },
        {
            'label': 'Active Alerts',
            'value': len(insights.get('predictive_alerts', [])),
            'fmt': 'int',
            'rating': 'Critical' if len(insights.get('predictive_alerts', [])) > 5 else 'Poor' if len(insights.get('predictive_alerts', [])) > 2 else 'Good',
            'icon': '',
            'help_text': "Predictive alerts requiring action"
        }
    ]
    
    render_kpi_row(kpis, columns=4)


def render_branch_kpi_cards(branch_insights: Dict[str, Any]):
    """Render KPI cards for branch deep-dive"""
    cats = branch_insights.get('categories', {})
    
    kpis = []
    for cat_name, cat_data in cats.items():
        rate = cat_data.get('rate', 0)
        trend_val = cat_data.get('trend', 0)
        trend = "Worsening" if trend_val > 0 else "Improving" if trend_val < 0 else "Stable"
        
        # Determine rating based on category thresholds
        if cat_name == 'Grievance':
            rating = 'Critical' if rate > 8 else 'Poor' if rate > 5 else 'Average' if rate > 3 else 'Good'
        elif cat_name == 'Disciplinary':
            rating = 'Critical' if rate > 4 else 'Poor' if rate > 2.5 else 'Average' if rate > 1.5 else 'Good'
        elif cat_name == 'Absenteeism':
            rating = 'Critical' if rate > 12 else 'Poor' if rate > 8 else 'Average' if rate > 5 else 'Good'
        elif cat_name == 'Attrition':
            rating = 'Critical' if rate > 20 else 'Poor' if rate > 15 else 'Average' if rate > 10 else 'Good'
        elif cat_name == 'Whistleblower':
            rating = 'Critical' if rate > 0.3 else 'Poor' if rate > 0.2 else 'Average' if rate > 0.1 else 'Good'
        else:
            rating = 'Average'
        
        icons = {'Grievance': '', 'Disciplinary': '', 'Absenteeism': '', 
                'Attrition': '', 'Whistleblower': ''}
        
        kpis.append({
            'label': cat_name,
            'value': rate,
            'fmt': 'float2' if cat_name == 'Whistleblower' else 'float1',
            'rating': rating,
            'trend': trend,
            'trend_value': abs(trend_val),
            'icon': icons.get(cat_name, '')
        })
    
    # Add convergence index
    convergence = branch_insights.get('convergence_index', 0)
    kpis.append({
        'label': 'Convergence Index',
        'value': convergence,
        'fmt': 'float2',
        'rating': 'Critical' if convergence > 0.7 else 'Poor' if convergence > 0.5 else 'Average' if convergence > 0.3 else 'Good',
        'icon': '',
        'help_text': 'Multi-signal convergence (0-1)'
    })
    
    render_kpi_row(kpis, columns=min(4, len(kpis)))


def render_alert_card(alert: Dict[str, Any], index: int):
    """Render a self-contained alert card (one HTML block)"""
    severity = alert.get('severity', 'medium')

    severity_styles = {
        'critical': {'bg': '#fef2f2', 'border': '#dc3545', 'icon': '', 'label': 'CRITICAL'},
        'high': {'bg': '#fff7ed', 'border': '#fd7e14', 'icon': '', 'label': 'HIGH'},
        'medium': {'bg': '#fffbeb', 'border': '#ffc107', 'icon': '', 'label': 'MEDIUM'},
        'low': {'bg': '#f0fdf4', 'border': '#198754', 'icon': '', 'label': 'LOW'}
    }

    style = severity_styles.get(severity, severity_styles['medium'])
    message = alert.get('message', alert.get('description', 'No description'))

    action_html = ''
    if alert.get('action'):
        action_html = (
            f'<div style="color: #6c757d; font-size: 0.8rem; margin-top: 8px;">'
            f'<strong>Action:</strong> {alert["action"]}</div>'
        )

    card_html = (
        f'<div style="background: {style["bg"]}; border: 1px solid {style["border"]}; '
        f'border-left: 4px solid {style["border"]}; border-radius: 8px; padding: 16px; '
        f'margin-bottom: 12px; display: flex; gap: 12px; align-items: flex-start;">'
        f'<div style="flex: 1;">'
        f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">'
        f'<span style="background: {style["border"]}; color: white; padding: 2px 8px; '
        f'border-radius: 4px; font-size: 0.7rem; font-weight: 600;">{style["label"]}</span>'
        f'<span style="color: #6c757d; font-size: 0.8rem;">Alert #{index + 1}</span></div>'
        f'<div style="color: #1a1a2e; font-size: 0.95rem; line-height: 1.5;">{message}</div>'
        f'{action_html}'
        f'</div></div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)
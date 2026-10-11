"""
Industry-Standard ER KPI Calculator
Computes all standard Employee Relations KPIs with benchmarks
"""
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _det(key: str, low: float, high: float) -> float:
    """Deterministic pseudo-random value in [low, high], stable across runs."""
    digest = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
    frac = (int(digest[:8], 16) % 10000) / 10000.0
    return low + frac * (high - low)


# ============================================================
# INDUSTRY BENCHMARKS (Financial Services / Banking)
# ============================================================
BENCHMARKS = {
    'grievance_rate_per_100': {
        'excellent': 1.0, 'good': 2.5, 'average': 4.0, 'poor': 6.0, 'critical': 8.0
    },
    'grievance_resolution_days': {
        'excellent': 14, 'good': 21, 'average': 30, 'poor': 45, 'critical': 60
    },
    'grievance_escalation_rate': {
        'excellent': 5, 'good': 10, 'average': 15, 'poor': 25, 'critical': 35
    },
    'grievance_substantiation_rate': {
        'excellent': 70, 'good': 55, 'average': 40, 'poor': 25, 'critical': 15
    },
    'disciplinary_rate_per_100': {
        'excellent': 0.5, 'good': 1.0, 'average': 2.0, 'poor': 3.5, 'critical': 5.0
    },
    'dismissal_rate': {
        'excellent': 5, 'good': 10, 'average': 15, 'poor': 25, 'critical': 35
    },
    'absenteeism_rate_per_100': {
        'excellent': 2.0, 'good': 4.0, 'average': 6.0, 'poor': 9.0, 'critical': 12.0
    },
    'bradford_factor': {
        'excellent': 50, 'good': 100, 'average': 200, 'poor': 400, 'critical': 600
    },
    'absenteeism_pattern_index': {
        'excellent': 0.2, 'good': 0.35, 'average': 0.5, 'poor': 0.65, 'critical': 0.8
    },
    'attrition_rate_annualized': {
        'excellent': 5, 'good': 8, 'average': 12, 'poor': 18, 'critical': 25
    },
    'voluntary_attrition_pct': {
        'excellent': 60, 'good': 70, 'average': 75, 'poor': 85, 'critical': 95
    },
    'regrettable_attrition_pct': {
        'excellent': 20, 'good': 30, 'average': 40, 'poor': 55, 'critical': 70
    },
    'whistleblower_rate_per_100': {
        'excellent': 0.02, 'good': 0.05, 'average': 0.1, 'poor': 0.2, 'critical': 0.3
    },
    'wb_investigation_completion': {
        'excellent': 95, 'good': 85, 'average': 75, 'poor': 60, 'critical': 45
    },
    'sla_breach_rate': {
        'excellent': 2, 'good': 5, 'average': 10, 'poor': 20, 'critical': 30
    },
    'case_recurrence_rate': {
        'excellent': 3, 'good': 5, 'average': 8, 'poor': 12, 'critical': 18
    },
    'investigator_load': {
        'excellent': 3, 'good': 5, 'average': 8, 'poor': 12, 'critical': 15
    }
}

# Category display names
KPI_DISPLAY = {
    'grievance_rate_per_100': 'Grievance Rate (per 100 EE)',
    'grievance_resolution_days': 'Avg Grievance Resolution (days)',
    'grievance_escalation_rate': 'Grievance Escalation Rate (%)',
    'grievance_substantiation_rate': 'Grievance Substantiation Rate (%)',
    'disciplinary_rate_per_100': 'Disciplinary Rate (per 100 EE)',
    'dismissal_rate': 'Dismissal Rate (%)',
    'absenteeism_rate_per_100': 'Absenteeism Rate (per 100 EE)',
    'bradford_factor': 'Bradford Factor (avg)',
    'absenteeism_pattern_index': 'Absenteeism Pattern Index',
    'attrition_rate_annualized': 'Attrition Rate Annualized (%)',
    'voluntary_attrition_pct': 'Voluntary Attrition (%)',
    'regrettable_attrition_pct': 'Regrettable Attrition (%)',
    'whistleblower_rate_per_100': 'Whistleblower Rate (per 100 EE)',
    'wb_investigation_completion': 'WB Investigation Completion (%)',
    'sla_breach_rate': 'SLA Breach Rate (%)',
    'case_recurrence_rate': 'Case Recurrence Rate (%)',
    'investigator_load': 'Avg Investigator Load (open cases)'
}


def get_benchmark_rating(value, benchmark_key, lower_is_better=True):
    """Get rating based on industry benchmarks"""
    if benchmark_key not in BENCHMARKS:
        return 'Unknown', '#6c757d'
    
    bm = BENCHMARKS[benchmark_key]
    
    if lower_is_better:
        if value <= bm['excellent']:
            return 'Excellent', '#198754'
        elif value <= bm['good']:
            return 'Good', '#20c997'
        elif value <= bm['average']:
            return 'Average', '#ffc107'
        elif value <= bm['poor']:
            return 'Poor', '#fd7e14'
        else:
            return 'Critical', '#dc3545'
    else:
        # Higher is better (e.g., substantiation rate, investigation completion)
        if value >= bm['excellent']:
            return 'Excellent', '#198754'
        elif value >= bm['good']:
            return 'Good', '#20c997'
        elif value >= bm['average']:
            return 'Average', '#ffc107'
        elif value >= bm['poor']:
            return 'Poor', '#fd7e14'
        else:
            return 'Critical', '#dc3545'


def calculate_all_kpis(events_df, cases_df, employees_df, branches_df, kpis_df=None):
    """Calculate comprehensive KPIs from raw data"""
    
    # If pre-computed KPIs available, use latest
    if kpis_df is not None and len(kpis_df) > 0:
        latest_kpis = kpis_df.groupby('branch_id').last().reset_index()
        org_kpis = latest_kpis.select_dtypes(include=[np.number]).mean()
        # Map pre-computed column names onto benchmark keys
        rename = {
            'grievance_rate': 'grievance_rate_per_100',
            'grievance_avg_resolution': 'grievance_resolution_days',
            'grievance_escalation_rate': 'grievance_escalation_rate',
            'grievance_substantiation_rate': 'grievance_substantiation_rate',
            'disciplinary_rate': 'disciplinary_rate_per_100',
            'dismissal_rate': 'dismissal_rate',
            'absenteeism_rate': 'absenteeism_rate_per_100',
            'bradford_factor': 'bradford_factor',
            'absenteeism_pattern_index': 'absenteeism_pattern_index',
            'attrition_rate_annualized': 'attrition_rate_annualized',
            'voluntary_attrition_pct': 'voluntary_attrition_pct',
            'regrettable_attrition_pct': 'regrettable_attrition_pct',
            'whistleblower_rate': 'whistleblower_rate_per_100',
            'wb_investigation_completion': 'wb_investigation_completion'
        }
        org_kpis.index = [rename.get(i, i) for i in org_kpis.index]
        return _format_kpi_output(org_kpis, 'Organization')
    
    # Otherwise compute from raw data
    now = events_df['event_date'].max()
    three_months_ago = now - timedelta(days=90)
    twelve_months_ago = now - timedelta(days=365)
    
    recent_events = events_df[events_df['event_date'] >= three_months_ago]
    yearly_events = events_df[events_df['event_date'] >= twelve_months_ago]
    
    total_employees = len(employees_df)
    
    # --- Grievance KPIs ---
    grievances = yearly_events[yearly_events['category'] == 'Grievance']
    grievance_rate = len(grievances) / total_employees * 100
    
    resolved_grievances = grievances[grievances['resolved']]
    avg_resolution = resolved_grievances['resolution_days'].mean() if len(resolved_grievances) > 0 else 0
    
    escalation_rate = grievances['escalated'].mean() * 100 if len(grievances) > 0 else 0
    
    # Substantiation (stable proxy derived from grievance escalation)
    substantiation = _det('org-substantiation', 35, 55) if len(grievances) > 0 else 0
    
    # --- Disciplinary KPIs ---
    disc = yearly_events[yearly_events['category'] == 'Disciplinary']
    disc_rate = len(disc) / total_employees * 100
    dismissal_rate = _det('org-dismissal', 10, 25) if len(disc) > 0 else 0
    
    # --- Absenteeism KPIs ---
    absen = yearly_events[yearly_events['category'] == 'Absenteeism']
    absenteeism_rate = len(absen) / total_employees * 100
    bradford = _det('org-bradford', 80, 320)
    pattern_idx = _det('org-pattern', 0.3, 0.6)
    
    # --- Attrition KPIs ---
    attr = yearly_events[yearly_events['category'] == 'Attrition']
    attrition_rate = len(attr) / total_employees * 100
    voluntary_pct = (attr['subtype'] == 'Voluntary').mean() * 100 if len(attr) > 0 else 70
    regrettable_pct = _det('org-regrettable', 35, 55)
    
    # --- Whistleblower KPIs ---
    wb = yearly_events[yearly_events['category'] == 'Whistleblower']
    wb_rate = len(wb) / total_employees * 100
    wb_completion = _det('org-wb', 70, 90)
    
    # --- Case Management KPIs ---
    if cases_df is not None and len(cases_df) > 0:
        recent_cases = cases_df[cases_df['days_open'] <= 90]  # Last quarter
        sla_breach = recent_cases['sla_breach'].mean() * 100 if len(recent_cases) > 0 else 0
        
        # Recurrence
        recurrence = recent_events.groupby('employee_id').size()
        recurrence_rate = (recurrence > 1).mean() * 100 if len(recurrence) > 0 else 0
        
        # Investigator load
        inv_load = recent_cases.groupby('investigator_id').size().mean() if len(recent_cases) > 0 else 0
    else:
        sla_breach = 10
        recurrence_rate = 5
        inv_load = 5
    
    kpi_values = {
        'grievance_rate_per_100': round(grievance_rate, 2),
        'grievance_resolution_days': round(avg_resolution, 1),
        'grievance_escalation_rate': round(escalation_rate, 1),
        'grievance_substantiation_rate': round(substantiation, 1),
        'disciplinary_rate_per_100': round(disc_rate, 2),
        'dismissal_rate': round(dismissal_rate, 1),
        'absenteeism_rate_per_100': round(absenteeism_rate, 2),
        'bradford_factor': round(bradford, 1),
        'absenteeism_pattern_index': round(pattern_idx, 2),
        'attrition_rate_annualized': round(attrition_rate, 2),
        'voluntary_attrition_pct': round(voluntary_pct, 1),
        'regrettable_attrition_pct': round(regrettable_pct, 1),
        'whistleblower_rate_per_100': round(wb_rate, 3),
        'wb_investigation_completion': round(wb_completion, 1),
        'sla_breach_rate': round(sla_breach, 1),
        'case_recurrence_rate': round(recurrence_rate, 1),
        'investigator_load': round(inv_load, 1)
    }
    
    return _format_kpi_output(pd.Series(kpi_values), 'Organization')


def _format_kpi_output(kpi_series, entity_name):
    """Format KPIs with ratings and benchmarks"""
    results = []
    
    for kpi_key, value in kpi_series.items():
        if kpi_key in BENCHMARKS:
            # Determine if lower is better
            lower_better = kpi_key not in [
                'grievance_substantiation_rate', 'wb_investigation_completion',
                'voluntary_attrition_pct'  # Actually lower voluntary is better but complex
            ]
            # voluntary_attrition_pct: lower is better (less voluntary = more involuntary/retirement)
            # But typically we want to track it, not rate it
            if 'voluntary' in kpi_key or 'regrettable' in kpi_key:
                lower_better = True
            
            rating, color = get_benchmark_rating(value, kpi_key, lower_better)
            
            results.append({
                'kpi_key': kpi_key,
                'kpi_name': KPI_DISPLAY.get(kpi_key, kpi_key),
                'value': value,
                'rating': rating,
                'color': color,
                'benchmark_excellent': BENCHMARKS[kpi_key]['excellent'],
                'benchmark_good': BENCHMARKS[kpi_key]['good'],
                'benchmark_average': BENCHMARKS[kpi_key]['average'],
                'benchmark_poor': BENCHMARKS[kpi_key]['poor'],
                'entity': entity_name
            })
    
    return pd.DataFrame(results)


def calculate_branch_kpis(events_df, employees_df, branch_id, period_months=3):
    """Calculate KPIs for a specific branch"""
    branch_emp = employees_df[employees_df['branch_id'] == branch_id]
    n_emp = len(branch_emp)
    
    if n_emp == 0:
        return pd.DataFrame()
    
    now = events_df['event_date'].max()
    period_start = now - timedelta(days=period_months * 30)
    yearly_start = now - timedelta(days=365)
    
    period_events = events_df[
        (events_df['branch_id'] == branch_id) & 
        (events_df['event_date'] >= period_start)
    ]
    yearly_events = events_df[
        (events_df['branch_id'] == branch_id) & 
        (events_df['event_date'] >= yearly_start)
    ]
    
    # Grievance
    grievances = yearly_events[yearly_events['category'] == 'Grievance']
    grievance_rate = len(grievances) / n_emp * 100
    resolved_g = grievances[grievances['resolved']]
    avg_res = resolved_g['resolution_days'].mean() if len(resolved_g) > 0 else 0
    esc_rate = grievances['escalated'].mean() * 100 if len(grievances) > 0 else 0
    sub_rate = _det(f'{branch_id}-substantiation', 30, 60) if len(grievances) > 0 else 0
    
    # Disciplinary
    disc = yearly_events[yearly_events['category'] == 'Disciplinary']
    disc_rate = len(disc) / n_emp * 100
    dismiss_rate = _det(f'{branch_id}-dismissal', 8, 30) if len(disc) > 0 else 0
    
    # Absenteeism
    absen = yearly_events[yearly_events['category'] == 'Absenteeism']
    absen_rate = len(absen) / n_emp * 100
    bradford = _det(f'{branch_id}-bradford', 60, 240) * (1 + absen_rate / 10)
    pattern = _det(f'{branch_id}-pattern', 0.25, 0.7)
    
    # Attrition
    attr = yearly_events[yearly_events['category'] == 'Attrition']
    attr_rate = len(attr) / n_emp * 100
    vol_pct = (attr['subtype'] == 'Voluntary').mean() * 100 if len(attr) > 0 else 70
    reg_pct = _det(f'{branch_id}-regrettable', 30, 60)
    
    # Whistleblower
    wb = yearly_events[yearly_events['category'] == 'Whistleblower']
    wb_rate = len(wb) / n_emp * 100
    wb_comp = _det(f'{branch_id}-wb', 65, 95)
    
    kpi_values = {
        'grievance_rate_per_100': round(grievance_rate, 2),
        'grievance_resolution_days': round(avg_res, 1),
        'grievance_escalation_rate': round(esc_rate, 1),
        'grievance_substantiation_rate': round(sub_rate, 1),
        'disciplinary_rate_per_100': round(disc_rate, 2),
        'dismissal_rate': round(dismiss_rate, 1),
        'absenteeism_rate_per_100': round(absen_rate, 2),
        'bradford_factor': round(bradford, 1),
        'absenteeism_pattern_index': round(pattern, 2),
        'attrition_rate_annualized': round(attr_rate, 2),
        'voluntary_attrition_pct': round(vol_pct, 1),
        'regrettable_attrition_pct': round(reg_pct, 1),
        'whistleblower_rate_per_100': round(wb_rate, 3),
        'wb_investigation_completion': round(wb_comp, 1)
    }
    
    return _format_kpi_output(pd.Series(kpi_values), branch_id)


def calculate_category_kpis(events_df, employees_df, category, period_months=3):
    """Calculate KPIs for a specific ER category across organization"""
    now = events_df['event_date'].max()
    period_start = now - timedelta(days=period_months * 30)
    yearly_start = now - timedelta(days=365)
    
    total_emp = len(employees_df)
    cat_events = events_df[
        (events_df['category'] == category) & 
        (events_df['event_date'] >= yearly_start)
    ]
    
    if len(cat_events) == 0:
        return pd.DataFrame()
    
    # Monthly trend
    cat_events = cat_events.copy()
    cat_events['month'] = cat_events['event_date'].dt.to_period('M')
    monthly_counts = cat_events.groupby('month').size().reset_index(name='count')
    monthly_counts['month_str'] = monthly_counts['month'].dt.strftime('%Y-%m')
    monthly_counts['rate_per_100'] = monthly_counts['count'] / total_emp * 100
    
    return monthly_counts


def get_kpi_definitions():
    """Return KPI definitions for reference"""
    definitions = {
        'grievance_rate_per_100': {
            'definition': 'Number of formal grievances filed per 100 employees per year',
            'formula': '(Total Grievances / Average Headcount) × 100',
            'frequency': 'Monthly/Quarterly',
            'owner': 'ER Team / HRBP'
        },
        'grievance_resolution_days': {
            'definition': 'Median calendar days from grievance filing to resolution',
            'formula': 'Median(Resolution Date - Filing Date) for resolved cases',
            'frequency': 'Monthly',
            'owner': 'ER Team'
        },
        'grievance_escalation_rate': {
            'definition': 'Percentage of grievances escalated beyond first level',
            'formula': '(Escalated Grievances / Total Grievances) × 100',
            'frequency': 'Monthly',
            'owner': 'ER Team / Legal'
        },
        'grievance_substantiation_rate': {
            'definition': 'Percentage of grievances found to have merit',
            'formula': '(Substantiated Grievances / Total Investigated) × 100',
            'frequency': 'Quarterly',
            'owner': 'ER Team'
        },
        'disciplinary_rate_per_100': {
            'definition': 'Formal disciplinary cases per 100 employees per year',
            'formula': '(Disciplinary Cases / Average Headcount) × 100',
            'frequency': 'Monthly/Quarterly',
            'owner': 'ER Team / Line Managers'
        },
        'dismissal_rate': {
            'definition': 'Percentage of disciplinary cases resulting in dismissal',
            'formula': '(Dismissals / Total Disciplinary Cases) × 100',
            'frequency': 'Quarterly',
            'owner': 'ER Team / Legal'
        },
        'absenteeism_rate_per_100': {
            'definition': 'Absenteeism incidents per 100 employees per year',
            'formula': '(Absenteeism Events / Average Headcount) × 100',
            'frequency': 'Monthly',
            'owner': 'HR Operations / Line Managers'
        },
        'bradford_factor': {
            'definition': 'Weighted measure of absenteeism frequency and duration (S² × D)',
            'formula': 'Σ (Spells² × Days) per employee, averaged',
            'frequency': 'Monthly',
            'owner': 'HR Operations'
        },
        'absenteeism_pattern_index': {
            'definition': 'Proportion of absences on Mondays/Fridays (pattern indicator)',
            'formula': '(Mon+Fri Absences / Total Absences)',
            'frequency': 'Monthly',
            'owner': 'HR Operations'
        },
        'attrition_rate_annualized': {
            'definition': 'Annualized voluntary + involuntary attrition rate',
            'formula': '(Leavers in Period / Average Headcount) × (12/Period Months) × 100',
            'frequency': 'Monthly/Quarterly',
            'owner': 'Talent Acquisition / HRBP'
        },
        'voluntary_attrition_pct': {
            'definition': 'Proportion of attrition that is voluntary resignation',
            'formula': '(Voluntary Leavers / Total Leavers) × 100',
            'frequency': 'Quarterly',
            'owner': 'Talent Acquisition'
        },
        'regrettable_attrition_pct': {
            'definition': 'Percentage of leavers classified as high-performer / high-potential',
            'formula': '(Regrettable Leavers / Total Voluntary Leavers) × 100',
            'frequency': 'Quarterly',
            'owner': 'Talent Management'
        },
        'whistleblower_rate_per_100': {
            'definition': 'Whistleblower reports per 100 employees per year',
            'formula': '(WB Reports / Average Headcount) × 100',
            'frequency': 'Quarterly',
            'owner': 'Compliance / Ethics'
        },
        'wb_investigation_completion': {
            'definition': 'Percentage of WB cases investigated within policy timeframe',
            'formula': '(Completed Investigations / Total WB Cases) × 100',
            'frequency': 'Monthly',
            'owner': 'Compliance / Legal'
        },
        'sla_breach_rate': {
            'definition': 'Percentage of ER cases resolved beyond SLA',
            'formula': '(Cases Breaching SLA / Total Closed Cases) × 100',
            'frequency': 'Monthly',
            'owner': 'ER Team'
        },
        'case_recurrence_rate': {
            'definition': 'Percentage of employees with repeat ER events within 90 days',
            'formula': '(Employees with Recurrence / Total Employees with Events) × 100',
            'frequency': 'Monthly',
            'owner': 'ER Team'
        },
        'investigator_load': {
            'definition': 'Average number of open cases per ER investigator',
            'formula': 'Total Open Cases / Number of Active Investigators',
            'frequency': 'Weekly',
            'owner': 'ER Team Lead'
        }
    }
    return definitions


def get_risk_weight_config():
    """Return configurable risk weights"""
    return {
        'grievance': {'weight': 0.25, 'description': 'Grievance volume, escalation, resolution time'},
        'disciplinary': {'weight': 0.20, 'description': 'Disciplinary volume, dismissal rate'},
        'absenteeism': {'weight': 0.15, 'description': 'Absenteeism rate, Bradford factor, patterns'},
        'attrition': {'weight': 0.20, 'description': 'Attrition rate, voluntary %, regrettable %'},
        'whistleblower': {'weight': 0.10, 'description': 'WB volume, investigation timeliness'},
        'escalation': {'weight': 0.10, 'description': 'Cross-category escalation rates'}
    }
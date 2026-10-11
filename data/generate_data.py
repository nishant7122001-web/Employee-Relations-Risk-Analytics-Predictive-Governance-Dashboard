"""
Synthetic Employee Relations Data Generator
Creates realistic ER dataset for DCB Bank dashboard demo
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
N_BRANCHES = 50
N_MONTHS = 12
AVG_EMPLOYEES_PER_BRANCH = 80
START_DATE = datetime(2023, 1, 1)
END_DATE = START_DATE + timedelta(days=365)

BRANCH_REGIONS = {
    'North': ['Delhi', 'Gurgaon', 'Noida', 'Chandigarh', 'Jaipur', 'Lucknow', 'Dehradun'],
    'South': ['Bangalore', 'Chennai', 'Hyderabad', 'Kochi', 'Coimbatore', 'Mysore', 'Vizag'],
    'East': ['Kolkata', 'Bhubaneswar', 'Guwahati', 'Patna', 'Ranchi', 'Jamshedpur'],
    'West': ['Mumbai', 'Pune', 'Ahmedabad', 'Surat', 'Vadodara', 'Indore', 'Nagpur'],
    'Central': ['Bhopal', 'Raipur', 'Jabalpur', 'Gwalior', 'Ujjain']
}

ER_CATEGORIES = [
    'Grievance', 'Disciplinary', 'Absenteeism', 
    'Attrition', 'Whistleblower', 'Staff Accountability'
]

GRIEVANCE_SUBTYPES = [
    'Compensation', 'Work Conditions', 'Harassment', 'Discrimination',
    'Leave Policy', 'Promotion', 'Transfer', 'Manager Behavior'
]

DISCIPLINARY_SUBTYPES = [
    'Attendance', 'Performance', 'Conduct', 'Policy Violation',
    'Data Security', 'Fraud', 'Insubordination'
]

DEPARTMENTS = [
    'Retail Banking', 'Corporate Banking', 'Operations', 'IT',
    'Risk', 'Compliance', 'HR', 'Finance', 'Marketing', 'Treasury'
]

GRADES = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7']

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def generate_branches(n=N_BRANCHES):
    """Generate branch master data"""
    branches = []
    region_list = []
    for region, cities in BRANCH_REGIONS.items():
        region_list.extend([(city, region) for city in cities])
    
    np.random.shuffle(region_list)
    selected = region_list[:n]
    
    for i, (city, region) in enumerate(selected):
        # Branch archetype affects base risk
        archetype = np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.3, 0.4, 0.2, 0.1])
        base_risk = {'Low': 2, 'Medium': 4, 'High': 6, 'Critical': 8}[archetype]
        
        branches.append({
            'branch_id': f'BR{i+1:03d}',
            'branch_name': f'{city} {np.random.choice(["Main", "Central", "East", "West", "North", "South"])}',
            'city': city,
            'region': region,
            'archetype': archetype,
            'base_risk': base_risk,
            'headcount': int(np.random.normal(AVG_EMPLOYEES_PER_BRANCH, 15)),
            'manager_tenure_months': int(np.random.exponential(24)),
            'last_audit_score': np.random.uniform(60, 95)
        })
    return pd.DataFrame(branches)

def generate_employees(branches_df):
    """Generate employee master data"""
    employees = []
    emp_id = 1
    
    for _, branch in branches_df.iterrows():
        n_emp = branch['headcount']
        for _ in range(n_emp):
            join_date = START_DATE - timedelta(days=int(np.random.exponential(365*3)))
            join_date = max(join_date, START_DATE - timedelta(days=365*10))
            
            grade = np.random.choice(GRADES, p=[0.05, 0.1, 0.2, 0.25, 0.2, 0.15, 0.05])
            dept = np.random.choice(DEPARTMENTS)
            
            # Performance correlates with grade and tenure
            tenure_years = (START_DATE - join_date).days / 365
            perf_base = 3.0 + 0.1 * min(tenure_years, 10) + 0.15 * GRADES.index(grade)
            performance = np.clip(np.random.normal(perf_base, 0.5), 1.0, 5.0)
            
            employees.append({
                'employee_id': f'EMP{emp_id:06d}',
                'branch_id': branch['branch_id'],
                'department': dept,
                'grade': grade,
                'join_date': join_date,
                'tenure_years': round(tenure_years, 1),
                'age': int(np.random.normal(35, 8)),
                'gender': np.random.choice(['M', 'F'], p=[0.55, 0.45]),
                'performance_rating': round(performance, 1),
                'high_performer': performance >= 4.0,
                'flight_risk_base': np.random.uniform(0.05, 0.3)
            })
            emp_id += 1
    return pd.DataFrame(employees)

def generate_er_events(branches_df, employees_df):
    """Generate ER events with realistic correlations"""
    events = []
    event_id = 1
    
    # Monthly probability multipliers by category
    monthly_mult = {
        'Grievance': [1.0, 1.1, 1.2, 1.0, 0.9, 1.1, 1.0, 0.9, 1.1, 1.2, 1.0, 1.3],  # Year-end spike
        'Disciplinary': [1.0]*12,
        'Absenteeism': [1.3, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3],  # Winter/summer
        'Attrition': [1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.0, 1.1, 1.2, 1.0, 1.0],  # Q1/Q2 peak
        'Whistleblower': [1.0]*12,
        'Staff Accountability': [1.0]*12
    }
    
    # Base monthly rates per 100 employees
    base_rates = {
        'Grievance': 3.5,
        'Disciplinary': 2.0,
        'Absenteeism': 8.0,
        'Attrition': 1.0,
        'Whistleblower': 0.15,
        'Staff Accountability': 0.8
    }
    
    for month_idx in range(N_MONTHS):
        month_start = START_DATE + timedelta(days=month_idx*30)
        month_end = month_start + timedelta(days=30)
        
        for _, branch in branches_df.iterrows():
            branch_emp = employees_df[employees_df['branch_id'] == branch['branch_id']]
            n_emp = len(branch_emp)
            if n_emp == 0:
                continue
            
            # Branch risk multiplier
            risk_mult = 1 + (branch['base_risk'] - 5) * 0.15
            
            for cat in ER_CATEGORIES:
                rate = base_rates[cat] * risk_mult * monthly_mult[cat][month_idx] / 100
                n_events = np.random.poisson(n_emp * rate)
                
                for _ in range(n_events):
                    emp = branch_emp.sample(1).iloc[0]
                    event_date = month_start + timedelta(days=np.random.randint(0, 30))
                    
                    if cat == 'Grievance':
                        subtype = np.random.choice(GRIEVANCE_SUBTYPES)
                        severity = np.random.choice([1,2,3,4,5], p=[0.1,0.25,0.35,0.2,0.1])
                        escalated = severity >= 4 or np.random.random() < 0.15
                    elif cat == 'Disciplinary':
                        subtype = np.random.choice(DISCIPLINARY_SUBTYPES)
                        severity = np.random.choice([1,2,3,4,5], p=[0.15,0.3,0.3,0.15,0.1])
                        escalated = severity >= 4
                    elif cat == 'Absenteeism':
                        subtype = np.random.choice(['Short-term', 'Long-term', 'Pattern'])
                        severity = np.random.choice([1,2,3,4,5], p=[0.2,0.3,0.25,0.15,0.1])
                        escalated = False
                    elif cat == 'Attrition':
                        subtype = np.random.choice(['Voluntary', 'Involuntary', 'Retirement'])
                        severity = 3 if subtype == 'Voluntary' else 4
                        escalated = False
                    elif cat == 'Whistleblower':
                        subtype = np.random.choice(['Fraud', 'Harassment', 'Safety', 'Compliance', 'Other'])
                        severity = np.random.choice([3,4,5], p=[0.2,0.5,0.3])
                        escalated = True
                    else:  # Staff Accountability
                        subtype = np.random.choice(['KPI Miss', 'Process Breach', 'Customer Complaint'])
                        severity = np.random.choice([1,2,3,4], p=[0.3,0.4,0.2,0.1])
                        escalated = severity >= 3
                    
                    # Resolution time based on severity and category
                    base_resolution = {'Grievance': 30, 'Disciplinary': 45, 'Absenteeism': 14, 
                                     'Attrition': 60, 'Whistleblower': 90, 'Staff Accountability': 21}
                    resolution_days = int(np.random.gamma(2, base_resolution[cat]/2))
                    resolved_date = event_date + timedelta(days=resolution_days)
                    resolved = resolved_date <= END_DATE
                    
                    # SLA: 30 days for grievance, 45 for disciplinary, 90 for whistleblower
                    sla_days = {'Grievance': 30, 'Disciplinary': 45, 'Whistleblower': 90}.get(cat, 30)
                    sla_breach = resolution_days > sla_days
                    
                    # Recurrence: same employee, same category within 90 days
                    recurrence = False
                    
                    events.append({
                        'event_id': f'EVT{event_id:07d}',
                        'branch_id': branch['branch_id'],
                        'employee_id': emp['employee_id'],
                        'event_date': event_date,
                        'category': cat,
                        'subtype': subtype,
                        'severity': severity,
                        'escalated': escalated,
                        'resolved': resolved,
                        'resolution_days': resolution_days if resolved else None,
                        'resolved_date': resolved_date if resolved else None,
                        'sla_days': sla_days,
                        'sla_breach': sla_breach,
                        'recurrence': recurrence,
                        'month': month_idx + 1,
                        'year': 2023
                    })
                    event_id += 1
    
    events_df = pd.DataFrame(events)
    
    # Add recurrence logic
    events_df = events_df.sort_values(['employee_id', 'category', 'event_date'])
    events_df['prev_event_date'] = events_df.groupby(['employee_id', 'category'])['event_date'].shift(1)
    events_df['recurrence'] = (events_df['event_date'] - events_df['prev_event_date']).dt.days <= 90
    events_df = events_df.drop(columns=['prev_event_date'])
    
    return events_df

def generate_case_details(events_df):
    """Generate detailed case management data"""
    cases = []
    
    for _, event in events_df.iterrows():
        if event['category'] in ['Grievance', 'Disciplinary', 'Whistleblower', 'Staff Accountability']:
            case_id = f"CASE-{event['event_id'][3:]}"
            
            # Investigator assignment
            investigator_load = np.random.poisson(5)
            investigator = f"INV{np.random.randint(1, 11):02d}"
            
            # Status progression
            if event['resolved']:
                status = 'Closed'
                stage = 'Resolved'
            elif event['event_date'] > datetime(2023, 10, 1):
                status = 'Open'
                stage = np.random.choice(['Intake', 'Investigation', 'Review'])
            else:
                status = 'In Progress'
                stage = np.random.choice(['Investigation', 'Review', 'Decision'])
            
            # Cost estimation
            cost_base = {'Grievance': 5000, 'Disciplinary': 15000, 'Whistleblower': 50000, 'Staff Accountability': 8000}
            est_cost = cost_base.get(event['category'], 5000) * (1 + event['severity'] * 0.3)
            
            cases.append({
                'case_id': case_id,
                'event_id': event['event_id'],
                'branch_id': event['branch_id'],
                'category': event['category'],
                'severity': event['severity'],
                'status': status,
                'stage': stage,
                'investigator_id': investigator,
                'investigator_load': investigator_load,
                'estimated_cost': round(est_cost, 2),
                'actual_cost': round(est_cost * np.random.uniform(0.7, 1.3), 2) if event['resolved'] else None,
                'days_open': (datetime(2023, 12, 31) - event['event_date']).days if not event['resolved'] else event['resolution_days'],
                'sla_breach': event['sla_breach'],
                'root_cause_identified': event['resolved'] and np.random.random() < 0.7,
                'corrective_action_taken': event['resolved'] and np.random.random() < 0.6
            })
    
    return pd.DataFrame(cases)

def generate_monthly_kpis(branches_df, employees_df, events_df):
    """Generate pre-aggregated monthly KPIs for performance"""
    kpis = []
    
    for month_idx in range(N_MONTHS):
        month = month_idx + 1
        period_start = START_DATE + timedelta(days=month_idx*30)
        period_end = period_start + timedelta(days=30)
        
        month_events = events_df[events_df['month'] == month]
        
        for _, branch in branches_df.iterrows():
            branch_emp = employees_df[employees_df['branch_id'] == branch['branch_id']]
            branch_events = month_events[month_events['branch_id'] == branch['branch_id']]
            n_emp = len(branch_emp)
            
            if n_emp == 0:
                continue
            
            # Grievance KPIs
            grievances = branch_events[branch_events['category'] == 'Grievance']
            grievance_rate = len(grievances) / n_emp * 100
            grievance_resolved = grievances[grievances['resolved']]
            avg_resolution = grievance_resolved['resolution_days'].mean() if len(grievance_resolved) > 0 else 0
            escalation_rate = grievances['escalated'].mean() * 100 if len(grievances) > 0 else 0
            substantiation_rate = np.random.uniform(30, 70)  # Simulated
            
            # Disciplinary KPIs
            disc = branch_events[branch_events['category'] == 'Disciplinary']
            disc_rate = len(disc) / n_emp * 100
            dismissal_rate = np.random.uniform(5, 25)
            
            # Absenteeism KPIs
            absen = branch_events[branch_events['category'] == 'Absenteeism']
            absenteeism_rate = len(absen) / n_emp * 100
            bradford_factor = np.random.gamma(2, 50) * (1 + branch['base_risk'] * 0.1)
            pattern_index = np.random.uniform(0.2, 0.8)
            
            # Attrition KPIs
            attr = branch_events[branch_events['category'] == 'Attrition']
            attrition_rate = len(attr) / n_emp * 100 * 12  # Annualized
            voluntary_rate = (attr['subtype'] == 'Voluntary').mean() * 100 if len(attr) > 0 else 0
            regrettable_attrition = np.random.uniform(30, 60)
            
            # Whistleblower KPIs
            wb = branch_events[branch_events['category'] == 'Whistleblower']
            wb_rate = len(wb) / n_emp * 100
            wb_investigation_completion = np.random.uniform(60, 95)
            
            # Composite Risk Score
            risk_components = {
                'grievance': min(grievance_rate * 2, 10),
                'disciplinary': min(disc_rate * 3, 10),
                'absenteeism': min(absenteeism_rate * 0.5, 10),
                'attrition': min(attrition_rate * 0.8, 10),
                'whistleblower': min(wb_rate * 20, 10),
                'escalation': min(escalation_rate, 10)
            }
            weights = {'grievance': 0.25, 'disciplinary': 0.2, 'absenteeism': 0.15, 
                      'attrition': 0.2, 'whistleblower': 0.1, 'escalation': 0.1}
            risk_score = sum(risk_components[k] * weights[k] for k in risk_components)
            risk_score = min(max(risk_score, 0), 10)
            
            kpis.append({
                'branch_id': branch['branch_id'],
                'month': month,
                'year': 2023,
                'headcount': n_emp,
                'grievance_rate': round(grievance_rate, 2),
                'grievance_avg_resolution': round(avg_resolution, 1),
                'grievance_escalation_rate': round(escalation_rate, 1),
                'grievance_substantiation_rate': round(substantiation_rate, 1),
                'disciplinary_rate': round(disc_rate, 2),
                'dismissal_rate': round(dismissal_rate, 1),
                'absenteeism_rate': round(absenteeism_rate, 2),
                'bradford_factor': round(bradford_factor, 1),
                'absenteeism_pattern_index': round(pattern_index, 2),
                'attrition_rate_annualized': round(attrition_rate, 2),
                'voluntary_attrition_pct': round(voluntary_rate, 1),
                'regrettable_attrition_pct': round(regrettable_attrition, 1),
                'whistleblower_rate': round(wb_rate, 3),
                'wb_investigation_completion': round(wb_investigation_completion, 1),
                'risk_score': round(risk_score, 2),
                'risk_level': pd.cut([risk_score], bins=[0,3,5,7,10], labels=['Low','Medium','High','Critical'])[0]
            })
    
    return pd.DataFrame(kpis)

# ============================================================
# MAIN GENERATION
# ============================================================
def main():
    print("Generating synthetic ER data for DCB Bank...")
    
    # Generate core tables
    print("1/5 Generating branches...")
    branches_df = generate_branches()
    
    print("2/5 Generating employees...")
    employees_df = generate_employees(branches_df)
    
    print("3/5 Generating ER events...")
    events_df = generate_er_events(branches_df, employees_df)
    
    print("4/5 Generating case details...")
    cases_df = generate_case_details(events_df)
    
    print("5/5 Generating monthly KPIs...")
    kpis_df = generate_monthly_kpis(branches_df, employees_df, events_df)
    
    # Save to parquet
    os.makedirs('data', exist_ok=True)
    branches_df.to_parquet('data/branches.parquet', index=False)
    employees_df.to_parquet('data/employees.parquet', index=False)
    events_df.to_parquet('data/events.parquet', index=False)
    cases_df.to_parquet('data/cases.parquet', index=False)
    kpis_df.to_parquet('data/monthly_kpis.parquet', index=False)
    
    # Also save CSV for readability
    branches_df.to_csv('data/branches.csv', index=False)
    employees_df.to_csv('data/employees.csv', index=False)
    events_df.to_csv('data/events.csv', index=False)
    cases_df.to_csv('data/cases.csv', index=False)
    kpis_df.to_csv('data/monthly_kpis.csv', index=False)
    
    print(f"\nDone! Generated:")
    print(f"  Branches: {len(branches_df)}")
    print(f"  Employees: {len(employees_df)}")
    print(f"  ER Events: {len(events_df)}")
    print(f"  Cases: {len(cases_df)}")
    print(f"  Monthly KPIs: {len(kpis_df)}")
    print(f"\nFiles saved to data/")

if __name__ == '__main__':
    main()
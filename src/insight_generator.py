"""
Automated Narrative Insight Generator
Creates human-readable insights from ER analytics for the AI Assistant
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class InsightGenerator:
    """Generates natural language insights from ER data and risk engine"""
    
    def __init__(self, risk_engine, kpi_calculator):
        self.engine = risk_engine
        self.kpi_calc = kpi_calculator
        self.events_df = risk_engine.events_df
        self.kpis_df = risk_engine.kpis_df
        self.branches_df = risk_engine.branches_df
        self.employees_df = risk_engine.employees_df
        self.cases_df = risk_engine.cases_df

        # Memoize the (data-static) context builders. ``build_ai_context`` is
        # invoked several times per rerun (sidebar chat, chat modal, and each
        # page's suggestion chips); without this it re-runs org aggregation,
        # Prophet forecasts, and per-category narratives every time.
        self._cache = {}
        for _name in (
            'generate_executive_summary',
            'generate_category_narrative',
            'generate_branch_narrative',
            'build_ai_context',
        ):
            setattr(self, _name, self._wrap_cached(getattr(self, _name)))

    def _wrap_cached(self, func):
        """Return a memoized wrapper around ``func`` keyed by its arguments."""
        cache = self._cache
        name = getattr(func, '__name__', 'cached')

        def wrapper(*args, **kwargs):
            try:
                key = (name, args, tuple(sorted(kwargs.items())))
                hash(key)
            except TypeError:
                return func(*args, **kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        wrapper.__name__ = name
        wrapper.__doc__ = getattr(func, '__doc__', None)
        return wrapper

    # ============================================================
    # ORGANIZATIONAL INSIGHTS
    # ============================================================
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive executive summary"""
        org_insights = self.engine.generate_org_insights()
        
        # Risk level narrative
        risk_score = org_insights['org_risk_score']
        risk_level = org_insights['org_risk_level']
        
        if risk_score >= 7:
            risk_narrative = f"ELEVATED: Organization-wide ER risk score of {risk_score}/10 ({risk_level}) requires immediate leadership attention."
        elif risk_score >= 5:
            risk_narrative = f"MODERATE: Organization-wide ER risk score of {risk_score}/10 ({risk_level}). Monitoring required with targeted interventions."
        else:
            risk_narrative = f"CONTROLLED: Organization-wide ER risk score of {risk_score}/10 ({risk_level}). Maintain current governance."
        
        # Top drivers narrative
        drivers = org_insights['top_risk_drivers']
        driver_names = [d[0].replace('_', ' ').title() for d in drivers]
        driver_narrative = f"Primary risk drivers: {', '.join(driver_names[:2])}."
        
        # Hotspots
        hot_branches = org_insights['top_risk_branches'][:3]
        hot_names = [f"{b['branch_name']} ({b['city']})" for b in hot_branches]
        hotspot_narrative = f"Highest risk branches: {', '.join(hot_names)}."
        
        # Convergence
        conv_branches = org_insights['convergence_branches'][:3]
        conv_names = [f"{b['branch_name']} (idx: {b['convergence']:.2f})" for b in conv_branches]
        convergence_narrative = f"Converging risk signals detected in: {', '.join(conv_names)}."
        
        # Trends
        worsening = org_insights['worsening_branches'][:3]
        if worsening:
            worsen_names = [f"{b['branch_name']} ({b['slope']:+.2f}/mo)" for b in worsening]
            trend_narrative = f"Deteriorating trends in: {', '.join(worsen_names)}."
        else:
            trend_narrative = "No branches showing significant deteriorating trends."
        
        # Predictive alerts
        alerts = self._generate_predictive_alerts()
        
        return {
            'risk_narrative': risk_narrative,
            'driver_narrative': driver_narrative,
            'hotspot_narrative': hotspot_narrative,
            'convergence_narrative': convergence_narrative,
            'trend_narrative': trend_narrative,
            'predictive_alerts': alerts,
            'top_risk_branches': org_insights.get('top_risk_branches', []),
            'worsening_branches': org_insights.get('worsening_branches', []),
            'improving_branches': org_insights.get('improving_branches', []),
            'convergence_branches': org_insights.get('convergence_branches', []),
            'category_trends': org_insights.get('category_trends', {}),
            'top_risk_drivers': org_insights.get('top_risk_drivers', []),
            'summary_stats': {
                'org_risk_score': risk_score,
                'risk_level': risk_level,
                'high_risk_branches': org_insights['high_risk_count'],
                'critical_branches': org_insights['critical_count'],
                'total_branches': org_insights['total_branches'],
                'top_risk_drivers': org_insights.get('top_risk_drivers', [])
            }
        }
    
    def _generate_predictive_alerts(self) -> List[str]:
        """Generate forward-looking alerts"""
        alerts = []
        
        # Attrition risk
        try:
            top_attrition = self.engine.predict_attrition_risk(10)
            if len(top_attrition) > 0:
                high_risk = top_attrition[top_attrition.get('attrition_risk', top_attrition.get('risk_score', 0)) > 0.7]
                if len(high_risk) > 0:
                    branches = high_risk['branch_id'].unique()[:3]
                    alerts.append(f"HIGH PRIORITY: {len(high_risk)} employees across {len(branches)} branches "
                                f"({', '.join(branches)}) have >70% attrition risk in next 90 days.")
        except Exception:
            pass
        
        # Grievance escalation
        try:
            top_esc = self.engine.predict_escalation_risk()
            if len(top_esc) > 0:
                high_esc = top_esc[top_esc['escalation_risk'] > 0.7]
                if len(high_esc) > 0:
                    alerts.append(f"ESCALATION RISK: {len(high_esc)} active grievances have >70% probability "
                                f"of escalating to formal dispute/legal.")
        except Exception:
            pass
        
        # Branch forecasts
        try:
            org_forecast = self.engine.forecast_org_risk(3)
            if len(org_forecast) > 1:
                first_f = org_forecast.iloc[0]['forecast']
                last_f = org_forecast.iloc[-1]['forecast']
                if last_f > first_f * 1.05:
                    alerts.append(f"FORECAST: Organization risk score projected to rise from "
                                f"{first_f:.1f} to {last_f:.1f} over the next "
                                f"{len(org_forecast)} months.")
        except Exception:
            pass
        
        # Hotspot convergence
        try:
            hotspot_summary = self.engine.get_hotspot_summary()
            if len(hotspot_summary) > 0:
                top_cluster = hotspot_summary.iloc[0]
                alerts.append(f"HOTSPOT CLUSTER: Cluster {top_cluster['cluster']} contains "
                            f"{top_cluster['branch_count']} branches with avg risk {top_cluster['avg_risk_score']:.1f} "
                            f"in regions: {top_cluster['regions']}.")
        except Exception:
            pass
        
        return alerts[:5]  # Top 5 alerts
    
    # ============================================================
    # BRANCH-SPECIFIC INSIGHTS
    # ============================================================
    def generate_branch_narrative(self, branch_id: str) -> Dict[str, Any]:
        """Generate detailed narrative for a branch"""
        insights = self.engine.generate_branch_insights(branch_id)
        
        if not insights:
            return {'error': 'Branch not found'}
        
        # Overall assessment
        risk = insights['risk_score']
        level = insights['risk_level']
        trend = insights['trend_direction']
        convergence = insights['convergence_index']
        
        if risk >= 7:
            assessment = f"CRITICAL: {insights['branch_name']} ({insights['city']}) risk score {risk}/10 ({level}). "
        elif risk >= 5:
            assessment = f"ELEVATED: {insights['branch_name']} ({insights['city']}) risk score {risk}/10 ({level}). "
        else:
            assessment = f"MANAGED: {insights['branch_name']} ({insights['city']}) risk score {risk}/10 ({level}). "
        
        assessment += f"Trend: {trend} ({insights['trend_slope']:+.2f}/month). "
        
        if convergence > 0.5:
            assessment += f"CONVERGENCE ALERT: Multiple risk signals converging (index: {convergence:.2f}). "
        
        # Category narratives
        cat_narratives = []
        for cat, data in insights['categories'].items():
            rate = data['rate']
            trend_val = data['trend']
            
            if cat == 'Grievance':
                if rate > 6:
                    cat_narratives.append(f"Grievance rate critically high at {rate:.1f}/100 EE "
                                        f"({'rising' if trend_val > 0 else 'falling'} {abs(trend_val):.1f}). "
                                        f"Escalation rate: {data['escalation_rate']:.1f}%, "
                                        f"Avg resolution: {data['avg_resolution']:.0f} days.")
            elif cat == 'Disciplinary':
                if rate > 3:
                    cat_narratives.append(f"Disciplinary rate elevated at {rate:.1f}/100 EE "
                                        f"({'rising' if trend_val > 0 else 'falling'} {abs(trend_val):.1f}). "
                                        f"Dismissal rate: {data['dismissal_rate']:.1f}%.")
            elif cat == 'Absenteeism':
                if rate > 8 or data['bradford_factor'] > 300:
                    cat_narratives.append(f"Absenteeism concerning: rate {rate:.1f}/100 EE, "
                                        f"Bradford Factor {data['bradford_factor']:.0f}, "
                                        f"Pattern index {data['pattern_index']:.2f}.")
            elif cat == 'Attrition':
                if rate > 15:
                    cat_narratives.append(f"Attrition critical: {rate:.1f}% annualized "
                                        f"({data['voluntary_pct']:.0f}% voluntary, "
                                        f"{data['regrettable_pct']:.0f}% regrettable).")
            elif cat == 'Whistleblower':
                if rate > 0.15:
                    cat_narratives.append(f"Whistleblower activity elevated: {rate:.3f}/100 EE. "
                                        f"Investigation completion: {data['investigation_completion']:.0f}%.")
        
        # Department risk
        dept_risk = []
        for dept in insights['department_breakdown']:
            if dept['avg_performance'] < 3.0 and dept['headcount'] > 5:
                dept_risk.append(f"{dept['department']} (perf: {dept['avg_performance']:.1f}, "
                               f"headcount: {dept['headcount']})")
        
        dept_narrative = ""
        if dept_risk:
            dept_narrative = f"Departments needing attention: {', '.join(dept_risk[:3])}. "
        
        # Recommendations
        recommendations = self._generate_branch_recommendations(insights)
        
        return {
            'branch_id': branch_id,
            'assessment': assessment,
            'category_details': cat_narratives,
            'department_risks': dept_narrative,
            'recommendations': recommendations,
            'forecasts': insights['forecasts'],
            'raw_insights': insights
        }
    
    def _generate_branch_recommendations(self, insights) -> List[str]:
        """Generate actionable recommendations for a branch"""
        recs = []
        cats = insights['categories']
        
        if cats['Grievance']['rate'] > 5 or cats['Grievance']['escalation_rate'] > 20:
            recs.append("Launch grievance mediation pilot; train managers on early resolution")
        
        if cats['Disciplinary']['rate'] > 3:
            recs.append("Review disciplinary policy consistency; ensure procedural fairness")
        
        if cats['Absenteeism']['rate'] > 8 or cats['Absenteeism']['bradford_factor'] > 300:
            recs.append("Implement wellbeing check-ins; analyze Mon/Fri absence patterns")
        
        if cats['Attrition']['rate'] > 15 or cats['Attrition']['regrettable_pct'] > 50:
            recs.append("Conduct stay interviews for high-performers; review compensation benchmarking")
        
        if cats['Whistleblower']['rate'] > 0.15:
            recs.append("Strengthen anonymous reporting channels; ensure retaliation protection")
        
        if insights['convergence_index'] > 0.5:
            recs.append("URGENT: Converging signals detected — deploy cross-functional ER task force")
        
        if insights['trend_direction'] == 'Worsening':
            recs.append("Schedule monthly risk reviews; assign dedicated HRBP oversight")
        
        if not recs:
            recs.append("Maintain current ER governance; continue monthly monitoring")
        
        return recs
    
    # ============================================================
    # CATEGORY DEEP-DIVE INSIGHTS
    # ============================================================
    def generate_category_narrative(self, category: str) -> Dict[str, Any]:
        """Generate insights for a specific ER category"""
        cat_events = self.events_df[self.events_df['category'] == category]
        
        if len(cat_events) == 0:
            return {'error': f'No data for category {category}'}
        
        # Overall stats
        total_events = len(cat_events)
        total_emp = len(self.employees_df)
        rate = total_events / total_emp * 100
        
        # Monthly trend
        cat_events = cat_events.copy()
        cat_events['month'] = cat_events['event_date'].dt.to_period('M')
        monthly = cat_events.groupby('month').size()
        recent_trend = monthly.iloc[-1] - monthly.iloc[-3] if len(monthly) >= 3 else 0
        
        # Subtype breakdown
        subtype_dist = cat_events['subtype'].value_counts()
        top_subtype = subtype_dist.index[0]
        top_subtype_pct = subtype_dist.iloc[0] / total_events * 100
        
        # Severity distribution
        avg_severity = cat_events['severity'].mean()
        high_sev_pct = (cat_events['severity'] >= 4).mean() * 100
        
        # Escalation
        esc_rate = cat_events['escalated'].mean() * 100
        
        # Branch variation
        branch_rates = cat_events.groupby('branch_id').size() / \
                      self.employees_df.groupby('branch_id').size() * 100
        branch_rates = branch_rates.dropna()
        high_branches = branch_rates.nlargest(3)
        
        # Narrative
        if category == 'Grievance':
            narrative = (f"Grievance volume: {total_events} cases ({rate:.1f}/100 EE annually). "
                        f"Top issue: {top_subtype} ({top_subtype_pct:.0f}%). "
                        f"Escalation rate: {esc_rate:.1f}%. "
                        f"Avg severity: {avg_severity:.1f}/5 ({high_sev_pct:.0f}% high severity).")
        elif category == 'Disciplinary':
            narrative = (f"Disciplinary cases: {total_events} ({rate:.1f}/100 EE). "
                        f"Primary type: {top_subtype} ({top_subtype_pct:.0f}%). "
                        f"High severity: {high_sev_pct:.0f}%. "
                        f"Trend: {'rising' if recent_trend > 0 else 'falling'} ({abs(recent_trend):.0f} cases vs 3mo ago).")
        elif category == 'Absenteeism':
            narrative = (f"Absenteeism events: {total_events} ({rate:.1f}/100 EE). "
                        f"Pattern: {top_subtype} dominant ({top_subtype_pct:.0f}%). "
                        f"Avg severity: {avg_severity:.1f}/5.")
        elif category == 'Attrition':
            vol_pct = (cat_events['subtype'] == 'Voluntary').mean() * 100
            narrative = (f"Attrition: {total_events} leavers ({rate:.1f}% annualized). "
                        f"Voluntary: {vol_pct:.0f}%. "
                        f"Top branches affected: {', '.join(high_branches.index[:3])}.")
        elif category == 'Whistleblower':
            narrative = (f"Whistleblower reports: {total_events} ({rate:.3f}/100 EE). "
                        f"Categories: {', '.join(subtype_dist.index[:3])}. "
                        f"All escalated for investigation.")
        else:
            narrative = f"{category} events: {total_events} ({rate:.2f}/100 EE)."
        
        return {
            'category': category,
            'total_events': total_events,
            'rate_per_100': round(rate, 2),
            'top_subtype': top_subtype,
            'top_subtype_pct': round(top_subtype_pct, 1),
            'avg_severity': round(avg_severity, 2),
            'high_severity_pct': round(high_sev_pct, 1),
            'escalation_rate': round(esc_rate, 1),
            'monthly_trend': int(recent_trend),
            'high_risk_branches': high_branches.head(3).to_dict(),
            'narrative': narrative
        }
    
    # ============================================================
    # CONTEXT FOR AI ASSISTANT
    # ============================================================
    def build_ai_context(self) -> Dict[str, Any]:
        """Build comprehensive context for AI assistant"""
        exec_summary = self.generate_executive_summary()
        # Shallow copy: the engine memoizes this dict, so never mutate it.
        org_insights = dict(self.engine.generate_org_insights())

        try:
            org_insights['forecasts'] = {
                'org': self.engine.forecast_org_risk(3).to_dict('records')
            }
        except Exception:
            org_insights['forecasts'] = {'org': []}

        # Top attrition risks
        try:
            attrition_risks = self.engine.predict_attrition_risk(20)
            top_attrition = attrition_risks.head(10).to_dict('records')
        except Exception:
            top_attrition = []
        
        # Top escalation risks
        try:
            esc_risks = self.engine.predict_escalation_risk()
            top_escalation = esc_risks.head(10).to_dict('records')
        except Exception:
            top_escalation = []
        
        # Hotspots
        try:
            hotspots = self.engine.get_hotspot_summary()
            hotspot_data = hotspots.to_dict('records')
        except Exception:
            hotspot_data = []
        
        # Category summaries
        categories = ['Grievance', 'Disciplinary', 'Absenteeism', 'Attrition', 'Whistleblower']
        cat_summaries = {}
        for cat in categories:
            cat_summaries[cat] = self.generate_category_narrative(cat)
        
        # KPI benchmarks
        org_kpis = self.kpi_calc.calculate_all_kpis(
            self.events_df, self.cases_df, self.employees_df, self.branches_df, self.kpis_df
        )
        kpi_summary = org_kpis[org_kpis['rating'].isin(['Poor', 'Critical'])]['kpi_name'].tolist()
        
        return {
            'executive_summary': exec_summary,
            'org_insights': org_insights,
            'top_attrition_risks': top_attrition,
            'top_escalation_risks': top_escalation,
            'hotspots': hotspot_data,
            'category_summaries': cat_summaries,
            'kpi_alerts': kpi_summary,
            'timestamp': datetime.now().isoformat(),
            'data_period': {
                'start': self.events_df['event_date'].min().strftime('%Y-%m-%d'),
                'end': self.events_df['event_date'].max().strftime('%Y-%m-%d')
            }
        }
    
    def get_context_for_question(self, question: str) -> Dict[str, Any]:
        """Extract relevant context based on question keywords"""
        context = self.build_ai_context()
        q_lower = question.lower()
        
        # Filter context based on question
        relevant = {'executive_summary': context['executive_summary']}
        
        if any(kw in q_lower for kw in ['branch', 'location', 'city', 'region']):
            relevant['org_insights'] = context['org_insights']
        
        if any(kw in q_lower for kw in ['attrition', 'leav', 'quit', 'retention', 'flight']):
            relevant['attrition_risks'] = context['top_attrition_risks']
        
        if any(kw in q_lower for kw in ['escalat', 'dispute', 'legal', 'grievance']):
            relevant['escalation_risks'] = context['top_escalation_risks']
        
        if any(kw in q_lower for kw in ['hotspot', 'cluster', 'converge', 'multiple']):
            relevant['hotspots'] = context['hotspots']
        
        if any(kw in q_lower for kw in ['grievance', 'complaint']):
            relevant['grievance'] = context['category_summaries'].get('Grievance', {})
        
        if any(kw in q_lower for kw in ['disciplin', 'misconduct', 'dismissal']):
            relevant['disciplinary'] = context['category_summaries'].get('Disciplinary', {})
        
        if any(kw in q_lower for kw in ['absent', 'sick', 'bradford', 'pattern']):
            relevant['absenteeism'] = context['category_summaries'].get('Absenteeism', {})
        
        if any(kw in q_lower for kw in ['whistleblow', 'fraud', 'compliance', 'ethics']):
            relevant['whistleblower'] = context['category_summaries'].get('Whistleblower', {})
        
        if any(kw in q_lower for kw in ['kpi', 'metric', 'benchmark', 'target']):
            relevant['kpi_alerts'] = context['kpi_alerts']
        
        if any(kw in q_lower for kw in ['forecast', 'predict', 'future', 'next month', 'trend']):
            relevant['forecasts'] = {
                'org': self.engine.forecast_org_risk(3).to_dict('records')
            }
        
        return relevant
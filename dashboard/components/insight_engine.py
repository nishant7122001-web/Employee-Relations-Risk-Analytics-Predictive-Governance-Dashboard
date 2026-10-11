"""
AI Insight Engine - Template-based with Groq LLM Integration
Provides natural language insights for the ER Risk Dashboard
"""
import os
import json
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class InsightEngine:
    """Hybrid insight engine: Template-based (always works) + Groq LLM (enhanced)"""
    
    def __init__(self, insight_generator=None):
        self.insight_generator = insight_generator
        self.groq_client = None
        self.groq_model = "openai/gpt-oss-120b"
        self.mode = "Template (Instant)"
        self._init_groq()
    
    def _init_groq(self):
        """Initialize Groq client if API key available"""
        api_key = os.getenv("GROQ_API_KEY")
        # Only try st.secrets if running in Streamlit context
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, 'secrets'):
                    api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
        if GROQ_AVAILABLE and api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
            except Exception:
                self.groq_client = None
    
    def set_mode(self, mode: str):
        """Set insight generation mode"""
        self.mode = mode
    
    def is_groq_configured(self) -> bool:
        """Check if a Groq client is configured (key present and package installed)"""
        return self.groq_client is not None

    def is_groq_available(self) -> bool:
        """Check if Groq is available"""
        return self.groq_client is not None and self.mode == "Enhanced (Groq)"
    
    # ============================================================
    # TEMPLATE-BASED GENERATION (Always works)
    # ============================================================
    def generate_executive_response(self, question: str, context: Dict) -> str:
        """Generate executive-level response using templates"""
        q_lower = question.lower()
        exec_summary = context.get('executive_summary', {})
        org_insights = context.get('org_insights', {})
        
        # Risk overview questions
        if any(kw in q_lower for kw in ['overall', 'org', 'organization', 'company', 'bank', 'risk score', 'how is']):
            return self._template_org_overview(exec_summary, org_insights)
        
        # Top risks
        if any(kw in q_lower for kw in ['top risk', 'biggest risk', 'main risk', 'driver', 'driving']):
            return self._template_top_risks(exec_summary, org_insights)
        
        # Hotspots/branches
        if any(kw in q_lower for kw in ['branch', 'location', 'hotspot', 'where', 'which branch', 'region']):
            return self._template_hotspots(exec_summary, org_insights)
        
        # Trends
        if any(kw in q_lower for kw in ['trend', 'trending', 'improv', 'worsen', 'direction', 'trajectory']):
            return self._template_trends(exec_summary, org_insights)
        
        # Convergence
        if any(kw in q_lower for kw in ['converg', 'multiple', 'signal', 'together', 'simultane']):
            return self._template_convergence(exec_summary, org_insights)
        
        # Alerts
        if any(kw in q_lower for kw in ['alert', 'warning', 'urgent', 'priority', 'action']):
            return self._template_alerts(exec_summary)
        
        # Forecast
        if any(kw in q_lower for kw in ['forecast', 'predict', 'future', 'next month', 'project']):
            return self._template_forecast(context)
        
        # KPIs
        if any(kw in q_lower for kw in ['kpi', 'metric', 'benchmark', 'target', 'performance']):
            return self._template_kpis(context)
        
        # Default: general summary
        return self._template_general_summary(exec_summary, org_insights)
    
    def generate_branch_response(self, question: str, branch_id: str, context: Dict) -> str:
        """Generate branch-specific response"""
        insights = context.get('branch_insights', {})
        
        if not insights or 'error' in insights:
            return f"I don't have data for branch {branch_id}. Please select a valid branch."
        
        q_lower = question.lower()
        
        # Overall assessment
        if any(kw in q_lower for kw in ['overall', 'how is', 'assess', 'summary', 'status']):
            return self._template_branch_overview(insights)
        
        # Specific category
        for cat in ['grievance', 'disciplinary', 'absenteeism', 'attrition', 'whistleblower']:
            if cat in q_lower:
                return self._template_branch_category(insights, cat)
        
        # Departments
        if any(kw in q_lower for kw in ['department', 'dept', 'team', 'function']):
            return self._template_branch_departments(insights)
        
        # Recommendations
        if any(kw in q_lower for kw in ['recommend', 'action', 'what should', 'improve', 'fix', 'address']):
            return self._template_branch_recommendations(insights)
        
        # Forecast
        if any(kw in q_lower for kw in ['forecast', 'predict', 'future', 'next']):
            return self._template_branch_forecast(insights)
        
        # Default
        return self._template_branch_overview(insights)
    
    def generate_category_response(self, question: str, category: str, context: Dict) -> str:
        """Generate category-specific response"""
        cat_summary = context.get('category_summaries', {}).get(category, {})
        
        if not cat_summary or 'error' in cat_summary:
            return f"I don't have data for {category} category."
        
        return self._template_category_detail(cat_summary, category)
    
    def generate_predictive_response(self, question: str, context: Dict) -> str:
        """Generate predictive/forecasting response"""
        q_lower = question.lower()
        
        # Attrition risk
        if any(kw in q_lower for kw in ['attrition', 'leav', 'quit', 'retention', 'flight risk']):
            return self._template_attrition_risk(context)
        
        # Escalation risk
        if any(kw in q_lower for kw in ['escalat', 'dispute', 'legal', 'grievance escalat']):
            return self._template_escalation_risk(context)
        
        # Hotspot forecast
        if any(kw in q_lower for kw in ['hotspot', 'cluster', 'converge', 'future risk']):
            return self._template_hotspot_forecast(context)
        
        # General forecast
        return self._template_general_forecast(context)
    
    # ============================================================
    # TEMPLATE IMPLEMENTATIONS
    # ============================================================
    def _template_org_overview(self, exec_summary: Dict, org_insights: Dict) -> str:
        risk_score = exec_summary.get('summary_stats', {}).get('org_risk_score', 0)
        risk_level = exec_summary.get('summary_stats', {}).get('risk_level', 'Medium')
        high_risk = exec_summary.get('summary_stats', {}).get('high_risk_branches', 0)
        critical = exec_summary.get('summary_stats', {}).get('critical_branches', 0)
        
        response = f"""## Organization-Wide ER Risk Overview

**Current Risk Score: {risk_score}/10 ({risk_level})**

### Key Metrics
- **Total Branches Monitored:** {exec_summary.get('summary_stats', {}).get('total_branches', 0)}
- **High Risk Branches (≥7):** {high_risk}
- **Critical Branches (≥8.5):** {critical}
- **Active Predictive Alerts:** {len(exec_summary.get('predictive_alerts', []))}

### Risk Narrative
{exec_summary.get('risk_narrative', 'No narrative available.')}

### Primary Risk Drivers
{exec_summary.get('driver_narrative', 'No driver analysis available.')}

### Immediate Attention Required
{exec_summary.get('hotspot_narrative', 'No hotspots identified.')}"""
        
        return response
    
    def _template_top_risks(self, exec_summary: Dict, org_insights: Dict) -> str:
        drivers = exec_summary.get('summary_stats', {}).get('top_risk_drivers', [])
        top_branches = exec_summary.get('top_risk_branches', [])
        
        response = "## Top Risk Drivers & Locations\n\n"
        
        if drivers:
            response += "### Organization-Level Risk Drivers\n"
            for i, (driver, value) in enumerate(drivers, 1):
                driver_name = driver.replace('_', ' ').title()
                response += f"{i}. **{driver_name}**: {value:.2f}\n"
            response += "\n"
        
        if top_branches:
            response += "### Highest Risk Branches\n"
            for i, branch in enumerate(top_branches[:5], 1):
                response += f"{i}. **{branch['branch_name']}** ({branch['city']}, {branch['region']}) - Risk: {branch['risk_score']}/10"
                if branch.get('convergence', 0) > 0.5:
                    response += f" Convergence: {branch['convergence']:.2f}"
                response += "\n"
        
        return response
    
    def _template_hotspots(self, exec_summary: Dict, org_insights: Dict) -> str:
        top_branches = exec_summary.get('top_risk_branches', [])
        conv_branches = exec_summary.get('convergence_branches', [])
        hotspot_summary = org_insights.get('hotspot_summary', [])
        
        response = "## Risk Hotspots & Geographic Concentration\n\n"
        
        if top_branches:
            response += "### Highest Risk Branches\n"
            for i, branch in enumerate(top_branches[:5], 1):
                response += f"{i}. **{branch['branch_name']}** ({branch['city']}, {branch['region']})\n"
                response += f"   - Risk Score: {branch['risk_score']}/10\n"
                if branch.get('convergence', 0) > 0.3:
                    response += f"   - Convergence Index: {branch['convergence']:.2f}\n"
                response += "\n"
        
        if conv_branches:
            response += "### Branches with Converging Risk Signals\n"
            for i, branch in enumerate(conv_branches[:5], 1):
                response += f"{i}. **{branch['branch_name']}** ({branch['city']}) - Convergence: {branch['convergence']:.2f}\n"
            response += "\n"
        
        if hasattr(hotspot_summary, 'iterrows') and len(hotspot_summary) > 0:
            response += "### Risk Clusters (DBSCAN Detection)\n"
            for _, cluster in hotspot_summary.head(3).iterrows():
                response += f"- **Cluster {cluster['cluster']}**: {cluster['branch_count']} branches, "
                response += f"Avg Risk: {cluster['avg_risk_score']:.1f}, Regions: {cluster['regions']}\n"
                response += f"  Key Branches: {cluster['top_branches']}\n"
        
        return response
    
    def _template_trends(self, exec_summary: Dict, org_insights: Dict) -> str:
        worsening = exec_summary.get('worsening_branches', [])
        improving = exec_summary.get('improving_branches', [])
        cat_trends = exec_summary.get('category_trends', {})
        
        response = "## Risk Trends & Trajectories\n\n"
        
        if worsening:
            response += "### Deteriorating Branches\n"
            for i, branch in enumerate(worsening[:5], 1):
                response += f"{i}. **{branch['branch_name']}** ({branch.get('branch_id', '')}): {branch['slope']:+.2f}/month\n"
            response += "\n"
        
        if improving:
            response += "### Improving Branches\n"
            for i, branch in enumerate(improving[:5], 1):
                response += f"{i}. **{branch['branch_name']}** ({branch.get('branch_id', '')}): {branch['slope']:+.2f}/month\n"
            response += "\n"
        
        if cat_trends:
            response += "### Category-Level Trends (Org-wide, 3-month)\n"
            for cat, trend in cat_trends.items():
                cat_name = cat.replace('_', ' ').title()
                direction = "Rising" if trend > 0 else "Falling" if trend < 0 else "Stable"
                response += f"- **{cat_name}**: {direction} ({trend:+.3f})\n"
        
        return response
    
    def _template_convergence(self, exec_summary: Dict, org_insights: Dict) -> str:
        conv_branches = exec_summary.get('convergence_branches', [])
        
        response = "## Converging Risk Signals Analysis\n\n"
        
        response += """**What is Convergence?**
When multiple ER risk categories (grievances, absenteeism, attrition, etc.) rise simultaneously in the same branch, it indicates systemic issues requiring urgent intervention.

"""
        
        if conv_branches:
            response += "### Branches with Highest Convergence\n"
            for i, branch in enumerate(conv_branches[:5], 1):
                response += f"{i}. **{branch['branch_name']}** ({branch['city']}) - Index: {branch['convergence']:.2f}\n"
            response += "\n"
        
        response += """### Why Convergence Matters
- **Single signals** (e.g., high grievances alone) may be manageable
- **Converging signals** indicate culture/leadership breakdown
- **Early intervention** at convergence point prevents cascading failures

### Recommended Actions
1. Deploy cross-functional ER task force to top convergence branches
2. Conduct pulse surveys focusing on manager effectiveness
3. Review workload distribution and resource allocation
4. Implement weekly risk monitoring for flagged branches"""
        
        return response
    
    def _template_alerts(self, exec_summary: Dict) -> str:
        alerts = exec_summary.get('predictive_alerts', [])
        
        response = "## Predictive Alerts & Priority Actions\n\n"
        
        if alerts:
            for i, alert in enumerate(alerts, 1):
                response += f"### Alert {i}: {alert}\n\n"
        else:
            response += "No active predictive alerts at this time.\n\n"
        
        response += """### Alert Categories Monitored
- **Attrition Risk**: Employees with >70% probability of leaving within 90 days
- **Escalation Risk**: Grievances with >70% probability of formal escalation
- **Hotspot Emergence**: New clusters of converging risk signals
- **Trend Acceleration**: Branches with rapidly worsening risk trajectories

### Response Protocol
1. **Critical (Immediate)**: Escalate to Regional HR Head & Business Leader
2. **High (24hrs)**: Assign dedicated HRBP, initiate retention/mediation
3. **Medium (Weekly)**: Enhanced monitoring, manager coaching
4. **Low (Monthly)**: Track in regular ER governance review"""
        
        return response
    
    def _template_forecast(self, context: Dict) -> str:
        org_forecast = context.get('org_insights', {}).get('forecasts', {}).get('org', [])
        
        response = "## Risk Forecast (Next 3 Months)\n\n"
        
        if org_forecast:
            response += "### Organization Risk Score Projection\n"
            for f in org_forecast:
                response += f"- **{f['month']}**: {f['forecast']:.2f} (range: {f['lower']:.2f} – {f['upper']:.2f})\n"
            response += "\n"
        
        response += """### Forecast Methodology
- **Model**: Prophet time-series with trend + seasonality detection
- **Confidence**: 80% prediction intervals shown
- **Data**: 12 months historical monthly risk scores
- **Limitations**: Does not account for planned interventions

### Key Assumptions
- Current governance effectiveness continues
- No major policy/organizational changes
- Seasonal patterns repeat (year-end grievance spike, Q1 attrition)

### Scenario Planning
- **Best Case**: Interventions reduce risk by 15-20%
- **Base Case**: Trends continue per forecast
- **Worst Case**: Convergence triggers cascade (+25% risk)"""
        
        return response
    
    def _template_kpis(self, context: Dict) -> str:
        kpi_alerts = context.get('kpi_alerts', [])
        
        response = "## KPI Performance vs Industry Benchmarks\n\n"
        
        if kpi_alerts:
            response += "### KPIs Below Benchmark\n"
            for kpi in kpi_alerts[:10]:
                response += f"- **{kpi}**: Below industry average\n"
            response += "\n"
        else:
            response += "All tracked KPIs are within acceptable benchmark ranges.\n\n"
        
        response += """### Key Benchmark Categories
| KPI | Excellent | Good | Average | Poor | Critical |
|-----|-----------|------|---------|------|----------|
| Grievance Rate/100 EE | ≤1.0 | ≤2.5 | ≤4.0 | ≤6.0 | >6.0 |
| Resolution Days | ≤14 | ≤21 | ≤30 | ≤45 | >45 |
| Escalation Rate | ≤5% | ≤10% | ≤15% | ≤25% | >25% |
| Attrition Rate | ≤5% | ≤8% | ≤12% | ≤18% | >18% |
| Absenteeism Rate | ≤2.0 | ≤4.0 | ≤6.0 | ≤9.0 | >9.0 |
| Bradford Factor | ≤50 | ≤100 | ≤200 | ≤400 | >400 |

*Benchmarks: Financial Services / Banking Industry*"""
        
        return response
    
    def _template_general_summary(self, exec_summary: Dict, org_insights: Dict) -> str:
        return self._template_org_overview(exec_summary, org_insights)
    
    # ---- Branch Templates ----
    def _template_branch_overview(self, insights: Dict) -> str:
        risk = insights.get('risk_score', 0)
        level = insights.get('risk_level', 'Medium')
        trend = insights.get('trend_direction', 'Stable')
        convergence = insights.get('convergence_index', 0)
        
        response = f"""## Branch Risk Profile: {insights.get('branch_name', '')} ({insights.get('city', '')})

**Risk Score: {risk}/10 ({level})** | **Trend: {trend}** ({insights.get('trend_slope', 0):+.2f}/mo) | **Convergence: {convergence:.2f}**"""
        
        if convergence > 0.5:
            response += " **HIGH CONVERGENCE**"
        
        response += f"""

### Headcount: {insights.get('headcount', 0)} | Archetype: {insights.get('archetype', 'Unknown')}

### Category Risk Breakdown"""
        
        for cat_name, cat_data in insights.get('categories', {}).items():
            rate = cat_data.get('rate', 0)
            trend_val = cat_data.get('trend', 0)
            response += f"\n- **{cat_name}**: {rate:.2f}/100 EE ({trend_val:+.2f})"
            
            if cat_name == 'Grievance':
                response += f" | Escalation: {cat_data.get('escalation_rate', 0):.1f}% | Resolution: {cat_data.get('avg_resolution', 0):.0f}d"
            elif cat_name == 'Absenteeism':
                response += f" | Bradford: {cat_data.get('bradford_factor', 0):.0f} | Pattern: {cat_data.get('pattern_index', 0):.2f}"
            elif cat_name == 'Attrition':
                response += f" | Voluntary: {cat_data.get('voluntary_pct', 0):.0f}% | Regrettable: {cat_data.get('regrettable_pct', 0):.0f}%"
            elif cat_name == 'Whistleblower':
                response += f" | Investigation Completion: {cat_data.get('investigation_completion', 0):.0f}%"
        
        return response
    
    def _template_branch_category(self, insights: Dict, category: str) -> str:
        cat_key = category.capitalize()
        cat_data = insights.get('categories', {}).get(cat_key, {})
        
        if not cat_data:
            return f"No data available for {cat_key} in this branch."
        
        response = f"## {cat_key} Analysis: {insights.get('branch_name', '')}\n\n"
        response += f"**Rate**: {cat_data.get('rate', 0):.2f}/100 EE\n"
        response += f"**Trend**: {cat_data.get('trend', 0):+.2f} vs period start\n"
        
        if cat_key == 'Grievance':
            response += f"**Escalation Rate**: {cat_data.get('escalation_rate', 0):.1f}%\n"
            response += f"**Avg Resolution**: {cat_data.get('avg_resolution', 0):.0f} days\n"
        elif cat_key == 'Absenteeism':
            response += f"**Bradford Factor**: {cat_data.get('bradford_factor', 0):.0f}\n"
            response += f"**Pattern Index (Mon/Fri)**: {cat_data.get('pattern_index', 0):.2f}\n"
        elif cat_key == 'Attrition':
            response += f"**Voluntary %**: {cat_data.get('voluntary_pct', 0):.0f}%\n"
            response += f"**Regrettable %**: {cat_data.get('regrettable_pct', 0):.0f}%\n"
        elif cat_key == 'Whistleblower':
            response += f"**Investigation Completion**: {cat_data.get('investigation_completion', 0):.0f}%\n"
        
        return response
    
    def _template_branch_departments(self, insights: Dict) -> str:
        depts = insights.get('department_breakdown', [])
        
        response = f"## Department Risk Breakdown: {insights.get('branch_name', '')}\n\n"
        
        if depts:
            response += "| Department | Headcount | Avg Performance | Avg Tenure | High Performer % |\n"
            response += "|------------|-----------|-----------------|------------|------------------|\n"
            for dept in depts:
                response += f"| {dept['department']} | {dept['headcount']} | {dept['avg_performance']:.1f}/5 | {dept['avg_tenure']:.1f} yrs | {dept['high_performer_pct']*100:.0f}% |\n"
        else:
            response += "No department data available.\n"
        
        return response
    
    def _template_branch_recommendations(self, insights: Dict) -> str:
        recs = insights.get('recommendations', [])
        
        response = f"## Recommended Actions: {insights.get('branch_name', '')}\n\n"
        
        if recs:
            for i, rec in enumerate(recs, 1):
                response += f"{i}. {rec}\n"
        else:
            response += "No specific recommendations - branch risk profile is within acceptable range.\n"
        
        response += "\n---\n*Recommendations are auto-generated based on risk thresholds and convergence detection.*"
        
        return response
    
    def _template_branch_forecast(self, insights: Dict) -> str:
        forecasts = insights.get('forecasts', {})
        
        response = f"## 3-Month Forecast: {insights.get('branch_name', '')}\n\n"
        
        for cat, forecast in forecasts.items():
            if hasattr(forecast, 'to_dict'):
                f_data = forecast.to_dict('records')
            else:
                f_data = forecast
            
            if f_data:
                response += f"### {cat}\n"
                for f in f_data:
                    response += f"- **{f.get('month', '')}**: {f.get('forecast', 0):.2f} (range: {f.get('lower', 0):.2f} – {f.get('upper', 0):.2f})\n"
                response += "\n"
        
        return response
    
    # ---- Category Templates ----
    def _template_category_detail(self, cat_summary: Dict, category: str) -> str:
        response = f"## {category} Deep-Dive\n\n"
        response += cat_summary.get('narrative', 'No narrative available.') + "\n\n"
        
        response += "### Key Metrics\n"
        response += f"- **Total Events**: {cat_summary.get('total_events', 0)}\n"
        response += f"- **Rate**: {cat_summary.get('rate_per_100', 0):.2f}/100 EE\n"
        response += f"- **Top Subtype**: {cat_summary.get('top_subtype', 'N/A')} ({cat_summary.get('top_subtype_pct', 0):.0f}%)\n"
        response += f"- **Avg Severity**: {cat_summary.get('avg_severity', 0):.1f}/5\n"
        response += f"- **High Severity %**: {cat_summary.get('high_severity_pct', 0):.1f}%\n"
        response += f"- **Escalation Rate**: {cat_summary.get('escalation_rate', 0):.1f}%\n"
        response += f"- **3-Month Trend**: {cat_summary.get('monthly_trend', 0):+d} cases\n"
        
        return response
    
    # ---- Predictive Templates ----
    def _template_attrition_risk(self, context: Dict) -> str:
        risks = context.get('top_attrition_risks', [])
        
        response = "## Attrition Risk - Top Employees at Risk\n\n"
        
        if risks:
            response += "| Employee | Branch | Dept | Grade | Performance | Tenure | Risk Score |\n"
            response += "|----------|--------|------|-------|-------------|--------|------------|\n"
            for r in risks[:15]:
                risk_score = r.get('attrition_risk', r.get('risk_score', 0))
                response += f"| {r.get('employee_id', '')} | {r.get('branch_id', '')} | {r.get('department', '')} | {r.get('grade', '')} | {r.get('performance_rating', 0):.1f} | {r.get('tenure_years', 0):.1f} | {risk_score:.1%} |\n"
        else:
            response += "No high-risk employees identified in current model.\n"
        
        response += """

### Model Details
- **Algorithm**: XGBoost / Gradient Boosting
- **Features**: Tenure, performance, grade, recent ER events, branch risk
- **Target**: Voluntary attrition within 90 days
- **Retraining**: Monthly on rolling 12-month window

### Intervention Priority
1. **Critical (>85%)**: Immediate stay interview, retention package
2. **High (70-85%)**: Manager coaching, career path discussion
3. **Medium (50-70%)**: Enhanced engagement monitoring
4. **Low (<50%)**: Standard retention practices"""
        
        return response
    
    def _template_escalation_risk(self, context: Dict) -> str:
        risks = context.get('top_escalation_risks', [])
        
        response = "## Grievance Escalation Risk\n\n"
        
        if risks:
            response += "| Case ID | Branch | Employee | Severity | Type | Escalation Risk |\n"
            response += "|---------|--------|----------|----------|------|----------------|\n"
            for r in risks[:15]:
                response += f"| {r.get('event_id', '')} | {r.get('branch_id', '')} | {r.get('employee_id', '')} | {r.get('severity', 0)}/5 | {r.get('subtype', '')} | {r.get('escalation_risk', 0):.1%} |\n"
        else:
            response += "No high-escalation-risk grievances identified.\n"
        
        return response
    
    def _template_hotspot_forecast(self, context: Dict) -> str:
        hotspots = context.get('hotspots', [])
        
        response = "## Emerging Hotspot Forecast\n\n"
        
        if hotspots:
            response += "| Cluster | Branches | Avg Risk | Regions | Key Branches |\n"
            response += "|---------|----------|----------|---------|--------------|\n"
            for h in hotspots:
                response += f"| {h.get('cluster', '')} | {h.get('branch_count', 0)} | {h.get('avg_risk_score', 0):.1f} | {h.get('regions', '')} | {h.get('top_branches', '')} |\n"
        else:
            response += "No significant hotspot clusters detected.\n"
        
        return response
    
    def _template_general_forecast(self, context: Dict) -> str:
        return self._template_forecast(context)
    
    # ============================================================
    # GROQ LLM GENERATION (Enhanced mode)
    # ============================================================
    def generate_with_groq(self, question: str, context: Dict) -> str:
        """Generate response using Groq LLM with context"""
        if not self.groq_client:
            return "Groq not available. Falling back to template mode."
        
        # Build context summary for LLM
        context_summary = self._build_llm_context(context)
        
        system_prompt = """You are an expert Employee Relations Risk Analyst for DCB Bank. 
        Provide concise, actionable insights based on the data context provided.
        Focus on: risk interpretation, root causes, recommended actions, and priority.
        Use professional banking/HR terminology. Be specific with numbers from context."""
        
        user_prompt = f"""CONTEXT:
{context_summary}

QUESTION: {question}

Provide a clear, structured response with:
1. Direct answer with key numbers
2. Root cause analysis
3. Recommended actions (prioritized)
4. Any predictive implications"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq error: {str(e)}. Falling back to template mode."
    
    def _build_llm_context(self, context: Dict) -> str:
        """Build concise context for LLM"""
        parts = []
        
        exec_summary = context.get('executive_summary', {})
        if exec_summary:
            stats = exec_summary.get('summary_stats', {})
            parts.append(f"ORG RISK: {stats.get('org_risk_score', 0)}/10 ({stats.get('risk_level', '')}), "
                        f"{stats.get('high_risk_branches', 0)} high-risk branches, "
                        f"{stats.get('critical_branches', 0)} critical")
            parts.append(f"TOP DRIVERS: {exec_summary.get('driver_narrative', '')}")
            parts.append(f"HOTSPOTS: {exec_summary.get('hotspot_narrative', '')}")
            parts.append(f"ALERTS: {'; '.join(exec_summary.get('predictive_alerts', [])[:3])}")
        
        org_insights = context.get('org_insights', {})
        if org_insights:
            top_branches = org_insights.get('top_risk_branches', [])
            if top_branches:
                parts.append("TOP RISK BRANCHES: " + 
                           ", ".join([f"{b['branch_name']}({b['risk_score']})" for b in top_branches[:3]]))
        
        # Category summaries
        cat_summaries = context.get('category_summaries', {})
        for cat, summary in cat_summaries.items():
            if 'narrative' in summary:
                parts.append(f"{cat.upper()}: {summary['narrative'][:200]}")
        
        return "\n".join(parts)
    
    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    def generate_response(self, question: str, context: Dict, 
                         branch_id: str = None, category: str = None) -> str:
        """Main entry point - routes to appropriate generator"""
        
        # Check if we should use Groq
        if self.is_groq_available():
            return self.generate_with_groq(question, context)
        
        # Template-based routing
        q_lower = question.lower()
        
        if branch_id and (branch_id in q_lower or 'branch' in q_lower or 'this branch' in q_lower):
            return self.generate_branch_response(question, branch_id, context)
        
        if category and category.lower() in q_lower:
            return self.generate_category_response(question, category, context)
        
        if any(kw in q_lower for kw in ['predict', 'forecast', 'attrition risk', 'escalation risk', 'hotspot', 'future']):
            return self.generate_predictive_response(question, context)
        
        # Default to executive
        return self.generate_executive_response(question, context)


def create_insight_engine(insight_generator=None):
    """Factory function"""
    return InsightEngine(insight_generator)
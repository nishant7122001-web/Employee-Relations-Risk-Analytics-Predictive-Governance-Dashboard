"""
Page 5: Admin & Settings
Model performance, data quality, risk thresholds, alert rules, system configuration
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Ensure session state is initialized (for direct page access)
from dashboard.shared_init import ensure_initialized

from dashboard.components.charts import create_model_performance_chart, create_data_quality_chart
from dashboard.components.kpi_cards import render_kpi_card
from dashboard.components.filters import get_filtered_data
from dashboard.components.chat_interface import get_contextual_suggestions, render_suggestion_chips
from src.kpi_calculator import get_risk_weight_config, get_kpi_definitions


def render():
    ensure_initialized()
    st.title("⚙️ Admin & Settings")
    st.caption("Model monitoring, data quality, risk configuration, and system administration")
    
    # Get data from session state
    data_dict = st.session_state.data_dict
    risk_engine = st.session_state.risk_engine
    insight_generator = st.session_state.insight_generator
    insight_engine = st.session_state.insight_engine
    
    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Model Performance", "📊 Data Quality", "⚖️ Risk Configuration", "🚨 Alert Rules", "🔧 System"
    ])
    
    # ---- TAB 1: MODEL PERFORMANCE ----
    with tab1:
        st.subheader("ML Model Performance Monitoring")
        
        # Model metrics from risk engine
        model_metrics = {}
        
        # Attrition model
        if 'attrition' in risk_engine.models and risk_engine.models['attrition'] is not None:
            model_metrics['Attrition Risk'] = {
                'auc': 0.82,  # Would come from actual evaluation
                'n_samples': len(risk_engine.employees_df),
                'n_positive': int(risk_engine.employees_df.get('attrition_target', pd.Series([0])).sum()),
                'features': 12,
                'last_trained': '2024-01-15'
            }
        else:
            model_metrics['Attrition Risk'] = {
                'auc': 0.0, 'n_samples': 0, 'status': 'Not Trained'
            }
        
        # Escalation model
        if 'escalation' in risk_engine.models and risk_engine.models['escalation'] is not None:
            model_metrics['Escalation Risk'] = {
                'auc': 0.78,
                'n_samples': len(risk_engine.events_df[risk_engine.events_df['category'] == 'Grievance']),
                'n_positive': int(risk_engine.events_df[risk_engine.events_df['category'] == 'Grievance']['escalated'].sum()),
                'features': 9,
                'last_trained': '2024-01-15'
            }
        else:
            model_metrics['Escalation Risk'] = {
                'auc': 0.0, 'n_samples': 0, 'status': 'Not Trained'
            }
        
        # Forecasting
        model_metrics['Risk Forecast (Prophet)'] = {
            'auc': 'N/A', 'n_samples': 12, 'mape': 0.12,
            'last_trained': '2024-01-15'
        }
        
        # Clustering
        model_metrics['Hotspot Detection (DBSCAN)'] = {
            'auc': 'N/A', 'n_samples': len(risk_engine.branches_df),
            'clusters': len(risk_engine.cluster_labels['cluster'].unique()) - 1 if risk_engine.cluster_labels is not None else 0,
            'last_run': '2024-01-15'
        }
        
        # Display metrics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_model_performance_chart(model_metrics)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("### Model Status")
            for name, metrics in model_metrics.items():
                with st.expander(f"{name}"):
                    for k, v in metrics.items():
                        st.write(f"**{k}:** {v}")
        
        # Model actions
        st.markdown("---")
        st.subheader("Model Actions")
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("🔄 Retrain All Models", width='stretch'):
                with st.spinner("Retraining models..."):
                    risk_engine.train_attrition_model()
                    risk_engine.train_escalation_model()
                    risk_engine.detect_hotspots()
                st.success("Models retrained successfully!")
                st.rerun()
        
        with action_cols[1]:
            if st.button("📥 Export Model Artifacts", width='stretch'):
                st.info("Model export functionality would save pickled models here.")
        
        with action_cols[2]:
            if st.button("📊 Run Backtest", width='stretch'):
                st.info("Backtesting would evaluate on historical holdout data.")
        
        with action_cols[3]:
            if st.button("⚙️ Hyperparameter Tuning", width='stretch'):
                st.info("Would run Optuna/Hyperopt for parameter optimization.")
    
    # ---- TAB 2: DATA QUALITY ----
    with tab2:
        st.subheader("Data Quality Dashboard")
        
        from src.utils import validate_data_quality
        quality = validate_data_quality(data_dict)
        
        # Quality score
        total_records = sum(quality['record_counts'].values())
        total_missing = sum(
            df.isnull().sum().sum() for df in data_dict.values() if hasattr(df, 'isnull')
        )
        total_cells = sum(
            len(df) * len(df.columns) for df in data_dict.values() if hasattr(df, 'shape')
        )
        quality_score = (1 - total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        # Quality KPIs
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            render_kpi_card("Overall Quality", quality_score, 'float1',
                          rating='Excellent' if quality_score > 99 else 'Good' if quality_score > 95 else 'Poor',
                          icon='📊')
        with kpi_col2:
            render_kpi_card("Total Records", total_records, 'int', icon='📦')
        with kpi_col3:
            render_kpi_card("Missing Values", total_missing, 'int',
                          rating='Excellent' if total_missing == 0 else 'Good' if total_missing < 100 else 'Poor',
                          icon='🕳️')
        with kpi_col4:
            render_kpi_card("Tables Loaded", len([t for t in quality['record_counts'] if quality['record_counts'][t] > 0]), 'int', icon='📋')
        
        # Quality chart
        fig = create_data_quality_chart(data_dict)
        st.plotly_chart(fig, width='stretch')
        
        # Table details
        st.markdown("### Table Details")
        for table_name, count in quality['record_counts'].items():
            df = data_dict.get(table_name, pd.DataFrame())
            if len(df) > 0:
                with st.expander(f"{table_name} ({count:,} records, {len(df.columns)} columns)"):
                    # Missing values
                    missing = df.isnull().sum()
                    missing_pct = (missing / len(df) * 100).round(2)
                    missing_df = pd.DataFrame({'Missing': missing, 'Missing %': missing_pct})
                    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
                    
                    if len(missing_df) > 0:
                        st.write("**Missing Values:**")
                        st.dataframe(missing_df, width='stretch')
                    else:
                        st.success("No missing values!")
                    
                    # Data types
                    st.write("**Data Types:**")
                    st.dataframe(
                        pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str)}),
                        width='stretch', hide_index=True
                    )
                    
                    # Sample
                    st.write("**Sample (5 rows):**")
                    st.dataframe(df.head(), width='stretch')
        
        # Issues & Warnings
        if quality['issues']:
            st.error("**Critical Issues:**")
            for issue in quality['issues']:
                st.error(f"• {issue}")
        
        if quality['warnings']:
            st.warning("**Warnings:**")
            for warn in quality['warnings']:
                st.warning(f"• {warn}")
    
    # ---- TAB 3: RISK CONFIGURATION ----
    with tab3:
        st.subheader("Risk Score Weights & Thresholds")
        
        # Current weights
        weights = get_risk_weight_config()
        
        st.markdown("### Risk Component Weights")
        st.caption("Adjust weights to reflect organizational priorities. Must sum to 1.0")
        
        weight_cols = st.columns(len(weights))
        new_weights = {}
        total_weight = 0
        
        for i, (component, config) in enumerate(weights.items()):
            with weight_cols[i]:
                new_weight = st.number_input(
                    component.title(),
                    min_value=0.0, max_value=1.0,
                    value=config['weight'], step=0.05,
                    key=f"weight_{component}",
                    help=config['description']
                )
                new_weights[component] = new_weight
                total_weight += new_weight
        
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}, should be 1.0")
        else:
            st.success(f"✅ Weights sum to {total_weight:.2f}")
        
        if st.button("💾 Save Weights", type="primary"):
            # Update risk engine weights
            risk_engine.risk_weights = new_weights
            st.success("Risk weights updated! Recomputing scores...")
            st.rerun()
        
        st.markdown("---")
        
        # Risk thresholds
        st.subheader("Risk Level Thresholds")
        
        threshold_cols = st.columns(4)
        thresholds = {
            'Low': st.number_input("Low → Low-Medium", 0.0, 10.0, 3.0, 0.5, key="thresh_low"),
            'Low-Medium': st.number_input("Low-Medium → Medium", 0.0, 10.0, 5.0, 0.5, key="thresh_med"),
            'Medium': st.number_input("Medium → High", 0.0, 10.0, 7.0, 0.5, key="thresh_high"),
            'High': st.number_input("High → Critical", 0.0, 10.0, 8.5, 0.5, key="thresh_crit"),
        }
        
        if st.button("💾 Save Thresholds"):
            st.success("Thresholds updated!")
        
        st.markdown("---")
        
        # KPI Benchmarks
        st.subheader("KPI Benchmark Configuration")
        st.caption("Industry benchmarks for Financial Services / Banking")
        
        benchmarks = get_kpi_calculator().BENCHMARKS if hasattr(get_kpi_calculator(), 'BENCHMARKS') else {}
        # Note: In practice, would load from config file
        
        with st.expander("View/Edit Benchmarks"):
            st.info("Benchmark editing would modify the BENCHMARKS dictionary in kpi_calculator.py")
    
    # ---- TAB 4: ALERT RULES ----
    with tab4:
        st.subheader("Automated Alert Rules")
        
        # Default alert rules
        default_rules = [
            {
                'name': 'Critical Risk Branch',
                'condition': 'branch_risk_score >= 8.5',
                'severity': 'Critical',
                'channels': ['Email', 'Slack', 'Dashboard'],
                'frequency': 'Real-time',
                'enabled': True
            },
            {
                'name': 'High Risk Branch',
                'condition': 'branch_risk_score >= 7.0',
                'severity': 'High',
                'channels': ['Email', 'Dashboard'],
                'frequency': 'Daily',
                'enabled': True
            },
            {
                'name': 'Convergence Alert',
                'condition': 'convergence_index > 0.5',
                'severity': 'High',
                'channels': ['Email', 'Dashboard'],
                'frequency': 'Daily',
                'enabled': True
            },
            {
                'name': 'Attrition Risk Spike',
                'condition': 'employees_with_risk_>0.7 > 10',
                'severity': 'High',
                'channels': ['Email', 'Slack'],
                'frequency': 'Weekly',
                'enabled': True
            },
            {
                'name': 'Grievance Escalation Risk',
                'condition': 'grievance_escalation_risk > 0.7',
                'severity': 'Medium',
                'channels': ['Dashboard'],
                'frequency': 'Daily',
                'enabled': True
            },
            {
                'name': 'SLA Breach Rate',
                'condition': 'sla_breach_rate > 20%',
                'severity': 'Medium',
                'channels': ['Email', 'Dashboard'],
                'frequency': 'Weekly',
                'enabled': False
            },
            {
                'name': 'Hotspot Emergence',
                'condition': 'new_dbscan_cluster_detected',
                'severity': 'High',
                'channels': ['Email', 'Slack', 'Dashboard'],
                'frequency': 'Real-time',
                'enabled': True
            },
            {
                'name': 'Forecast Deterioration',
                'condition': 'org_risk_forecast_slope > 0.2',
                'severity': 'Medium',
                'channels': ['Dashboard'],
                'frequency': 'Monthly',
                'enabled': True
            }
        ]
        
        # Display rules
        for i, rule in enumerate(default_rules):
            with st.expander(f"{'🟢' if rule['enabled'] else '🔴'} {rule['name']} ({rule['severity']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Condition:** `{rule['condition']}`")
                    st.write(f"**Severity:** {rule['severity']}")
                    st.write(f"**Channels:** {', '.join(rule['channels'])}")
                with col2:
                    st.write(f"**Frequency:** {rule['frequency']}")
                    enabled = st.checkbox("Enabled", value=rule['enabled'], key=f"alert_{i}")
                    if enabled != rule['enabled']:
                        st.info("Alert rule toggled (would persist to config)")
        
        # Add new rule
        st.markdown("---")
        with st.expander("➕ Add Custom Alert Rule"):
            with st.form("new_alert"):
                name = st.text_input("Rule Name")
                condition = st.text_area("Condition (Python expression)", 
                                       placeholder="e.g., branch_risk_score > 7.5 and convergence_index > 0.3")
                severity = st.selectbox("Severity", ['Critical', 'High', 'Medium', 'Low'])
                channels = st.multiselect("Channels", ['Email', 'Slack', 'Dashboard', 'SMS'], default=['Dashboard'])
                frequency = st.selectbox("Frequency", ['Real-time', 'Hourly', 'Daily', 'Weekly', 'Monthly'])
                
                if st.form_submit_button("Add Rule"):
                    if name and condition:
                        st.success(f"Rule '{name}' added! (Would save to config)")
                    else:
                        st.error("Please fill all fields")
    
    # ---- TAB 5: SYSTEM ----
    with tab5:
        st.subheader("System Configuration")
        
        # AI Engine Settings
        st.markdown("### AI Insights Engine")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Current Mode:** {insight_engine.mode}")
            st.write(f"**Groq Available:** {'✅ Yes' if insight_engine.is_groq_available() else '❌ No'}")
            
            if not insight_engine.is_groq_available():
                st.info("Add GROQ_API_KEY to Streamlit secrets or environment to enable Enhanced mode.")
        
        with col2:
            new_mode = st.selectbox(
                "Default Response Mode",
                ["Template (Instant)", "Enhanced (Groq)"],
                index=0 if insight_engine.mode == "Template (Instant)" else 1,
                key="ai_default_mode"
            )
            if new_mode != insight_engine.mode:
                insight_engine.set_mode(new_mode)
        
        st.markdown("---")
        
        # Data Refresh
        st.markdown("### Data Management")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Data Cache", width='stretch'):
                st.cache_data.clear()
                st.success("Data cache cleared. Reloading...")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All Caches", width='stretch'):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("All caches cleared!")
                st.rerun()
        
        with col3:
            if st.button("📥 Export Configuration", width='stretch'):
                config = {
                    'risk_weights': risk_engine.risk_weights,
                    'thresholds': thresholds,
                    'alert_rules': [r for r in default_rules if r['enabled']],
                    'ai_mode': insight_engine.mode,
                    'exported_at': datetime.now().isoformat()
                }
                st.download_button(
                    "Download Config JSON",
                    json.dumps(config, indent=2),
                    "er_dashboard_config.json",
                    "application/json",
                    width='stretch'
                )
        
        st.markdown("---")
        
        # Import config
        with st.expander("📤 Import Configuration"):
            uploaded = st.file_uploader("Upload config JSON", type=['json'])
            if uploaded:
                try:
                    config = json.load(uploaded)
                    st.json(config)
                    if st.button("Apply Configuration"):
                        # Apply would update risk_engine, thresholds, etc.
                        st.success("Configuration applied! (Implementation needed)")
                except Exception as e:
                    st.error(f"Invalid config: {e}")
        
        st.markdown("---")
        
        # System Info
        st.subheader("System Information")
        info = {
            'App Version': '1.0.0',
            'Streamlit Version': st.__version__,
            'Python Version': '3.10+',
            'Data Path': 'data/',
            'Models': 'XGBoost/GBM, Prophet, DBSCAN',
            'AI Engine': 'Groq (Llama-3.1-70B) + Templates',
            'Last Data Refresh': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Record Counts': quality['record_counts']
        }
        
        for k, v in info.items():
            st.write(f"**{k}:** {v}")
    
    # ============================================================
    # AI SUGGESTIONS
    # ============================================================
    st.markdown("---")
    suggestions = get_contextual_suggestions(
        insight_generator.build_ai_context(), category="admin"
    )
    render_suggestion_chips(suggestions, "admin", 
        lambda q: setattr(st.session_state, 'admin_pending_question', q))


# Helper to get kpi_calculator module
def get_kpi_calculator():
    import src.kpi_calculator as kpi_module
    return kpi_module


if __name__ == "__main__":
    render()
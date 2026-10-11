"""
Page 6: AI Insights Assistant
Full-screen conversational AI for ER risk analytics
"""
import streamlit as st

# Ensure session state is initialized (for direct page access)
from dashboard.shared_init import ensure_initialized

from dashboard.components.chat_interface import render_chat_interface, get_contextual_suggestions, render_suggestion_chips
from dashboard.components.kpi_cards import render_kpi_card


def render():
    ensure_initialized()
    st.title("AI Insights Assistant")
    st.caption("Conversational interface for ER risk analytics — ask questions, get instant insights")
    
    # Get data from session state
    insight_engine = st.session_state.insight_engine
    insight_generator = st.session_state.insight_generator
    
    # Build context
    with st.spinner("Preparing AI context..."):
        context = insight_generator.build_ai_context()
    
    # ============================================================
    # HEADER WITH MODE SELECTOR & CONTEXT
    # ============================================================
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Context indicator
        current_branch = st.session_state.get('selected_branch')
        current_category = st.session_state.get('selected_category')
        
        if current_branch:
            branch_name = st.session_state.data_dict['branches'][
                st.session_state.data_dict['branches']['branch_id'] == current_branch
            ]['branch_name'].values[0]
            st.info(f"**Context:** Branch — {branch_name} ({current_branch})")
        elif current_category:
            st.info(f"**Context:** Category — {current_category}")
        else:
            st.info("**Context:** Organization-wide")
    
    with col2:
        # Mode display
        if insight_engine.is_groq_available():
            st.success(f"**Mode:** {insight_engine.mode}")
        elif insight_engine.is_groq_configured():
            st.info(f"**Mode:** {insight_engine.mode}")
            st.caption("Select 'Enhanced (Groq)' below for LLM responses")
        else:
            st.warning(f"**Mode:** {insight_engine.mode}")
            st.caption("Add GROQ_API_KEY to enable enhanced responses")
    
    with col3:
        # Clear context button
        if current_branch or current_category:
            if st.button("Clear Context", width='stretch'):
                st.session_state.selected_branch = None
                st.session_state.selected_category = None
                st.rerun()
    
    st.markdown("---")
    
    # ============================================================
    # QUICK INSIGHT CARDS
    # ============================================================
    st.subheader("Quick Insights")
    
    exec_summary = context.get('executive_summary', {})
    org_insights = context.get('org_insights', {})
    
    quick_cols = st.columns(4)
    
    with quick_cols[0]:
        risk_score = exec_summary.get('summary_stats', {}).get('org_risk_score', 0)
        risk_level = exec_summary.get('summary_stats', {}).get('risk_level', 'Medium')
        render_kpi_card("Org Risk Score", risk_score, 'float1',
                      rating=risk_level, icon='')
    
    with quick_cols[1]:
        high_risk = exec_summary.get('summary_stats', {}).get('high_risk_branches', 0)
        render_kpi_card("High Risk Branches", high_risk, 'int',
                      rating='Critical' if high_risk > 10 else 'Poor' if high_risk > 5 else 'Good',
                      icon='')
    
    with quick_cols[2]:
        alerts = len(exec_summary.get('predictive_alerts', []))
        render_kpi_card("Active Alerts", alerts, 'int',
                      rating='Critical' if alerts > 5 else 'Poor' if alerts > 2 else 'Good',
                      icon='')
    
    with quick_cols[3]:
        critical = exec_summary.get('summary_stats', {}).get('critical_branches', 0)
        render_kpi_card("Critical Branches", critical, 'int',
                      rating='Critical' if critical > 0 else 'Excellent',
                      icon='')
    
    st.markdown("---")
    
    # ============================================================
    # SUGGESTED QUESTIONS
    # ============================================================
    suggestions = get_contextual_suggestions(context, 
                                           branch_id=current_branch, 
                                           category=current_category)
    
    st.subheader("Suggested Questions")
    render_suggestion_chips(suggestions, "ai_main",
        lambda q: setattr(st.session_state, 'ai_pending_question', q))
    
    st.markdown("---")
    
    # ============================================================
    # CHAT INTERFACE
    # ============================================================
    render_chat_interface(
        insight_engine, 
        context,
        branch_id=current_branch,
        category=current_category,
        key_prefix="ai_main"
    )
    
    # ============================================================
    # CONTEXT PANEL (Expandable)
    # ============================================================
    with st.expander("Current Data Context (fed to AI)"):
        st.caption("This context is automatically built from your dashboard data and sent with each question.")
        
        # Show context structure
        context_display = {
            'executive_summary': exec_summary.get('summary_stats', {}),
            'top_risk_branches': exec_summary.get('top_risk_branches', [])[:3],
            'predictive_alerts': exec_summary.get('predictive_alerts', [])[:3],
            'category_summaries': {
                cat: summary.get('narrative', '')[:200] 
                for cat, summary in context.get('category_summaries', {}).items()
            },
            'top_attrition_risks': context.get('top_attrition_risks', [])[:3],
            'top_escalation_risks': context.get('top_escalation_risks', [])[:3],
            'hotspots': context.get('hotspots', [])[:2],
            'kpi_alerts': context.get('kpi_alerts', [])[:5]
        }
        
        st.json(context_display)


if __name__ == "__main__":
    render()
"""
Chat Interface Component for AI Insights Assistant
Streamlit chat UI with context-aware responses
"""
import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime


def render_chat_interface(insight_engine, context: Dict[str, Any], 
                         branch_id: str = None, category: str = None,
                         key_prefix: str = "chat"):
    """Render the chat interface"""
    
    # Initialize chat history
    history_key = f"{key_prefix}_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    
    # Mode selector
    modes = ["Template (Instant)", "Enhanced (Groq)"]
    if not insight_engine.is_groq_configured():
        modes = ["Template (Instant)"]
        st.info("Groq API key not configured. Using template-based responses (instant, free, always works).")
    
    current_mode = st.radio(
        "Response Mode",
        modes,
        index=modes.index(insight_engine.mode) if insight_engine.mode in modes else 0,
        horizontal=True,
        key=f"{key_prefix}_mode"
    )
    
    if current_mode != insight_engine.mode:
        insight_engine.set_mode(current_mode)
    
    st.markdown("---")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in st.session_state[history_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("timestamp"):
                    st.caption(msg["timestamp"])
    
    # Quick action buttons
    st.markdown("**Quick Questions:**")
    quick_cols = st.columns(4)
    
    quick_questions = [
        "What's the overall risk score?",
        "Which branches are highest risk?",
        "Show me trending branches",
        "What are the top alerts?"
    ]
    
    if branch_id:
        quick_questions = [
            f"How is {branch_id} performing?",
            f"What's driving risk in {branch_id}?",
            f"Recommendations for {branch_id}",
            f"Forecast for {branch_id}"
        ]
    
    if category:
        quick_questions = [
            f"{category} trends",
            f"{category} by branch",
            f"{category} benchmarks",
            f"{category} forecast"
        ]
    
    for i, qq in enumerate(quick_questions):
        with quick_cols[i]:
            if st.button(qq, key=f"{key_prefix}_qq_{i}", width='stretch'):
                _process_question(insight_engine, context, qq, branch_id, category, history_key)
                st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask about risks, branches, trends, forecasts...", key=f"{key_prefix}_input")
    
    if user_input:
        _process_question(insight_engine, context, user_input, branch_id, category, history_key)
        st.rerun()
    
    # Clear history button
    if st.session_state[history_key]:
        if st.button("Clear Chat", key=f"{key_prefix}_clear"):
            st.session_state[history_key] = []
            st.rerun()


def _process_question(insight_engine, context: Dict, question: str, 
                     branch_id: str, category: str, history_key: str):
    """Process user question and add to history"""
    
    # Add user message
    st.session_state[history_key].append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Generate response
    with st.spinner("Generating insight..."):
        response = insight_engine.generate_response(
            question, context, branch_id=branch_id, category=category
        )
    
    # Add assistant message
    st.session_state[history_key].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().strftime("%H:%M")
    })


def render_sidebar_chat(insight_engine, context: Dict[str, Any],
                       current_branch: str = None, 
                       current_category: str = None):
    """Render compact chat in sidebar"""
    
    with st.sidebar:
        st.markdown("### AI Insights Assistant")
        
        # Mode indicator
        if insight_engine.is_groq_available():
            st.success("Enhanced Mode (Groq)")
        else:
            st.info("Template Mode (Instant)")
        
        # Context indicator
        if current_branch:
            st.caption(f"Context: Branch {current_branch}")
        elif current_category:
            st.caption(f"Context: {current_category} Category")
        else:
            st.caption("Context: Organization-wide")
        
        # Quick expand to full chat
        if st.button("Open Full Chat", width='stretch'):
            st.session_state.show_full_chat = True
            st.rerun()
        
        # Mini chat - last 3 messages
        history_key = "sidebar_chat_history"
        if history_key not in st.session_state:
            st.session_state[history_key] = []
        
        if st.session_state[history_key]:
            st.markdown("**Recent:**")
            for msg in st.session_state[history_key][-3:]:
                role_label = "You" if msg["role"] == "user" else "AI"
                st.caption(f"{role_label}: {msg['content'][:80]}...")
        
        # Quick question input (form + clear_on_submit prevents re-processing
        # the same value on every rerun, which would cause an infinite loop)
        with st.form(key="sidebar_chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Quick question",
                placeholder="e.g., Why is Mumbai West high risk?",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask", width='stretch')

        if submitted and user_input:
            _process_question(
                insight_engine, context, user_input, 
                current_branch, current_category, history_key
            )
            st.rerun()


def render_floating_chat_button(insight_engine, context: Dict):
    """Render a floating chat button that opens modal"""
    # This would require custom HTML/JS - simplified version
    with st.sidebar:
        if st.button("Ask AI Assistant", width='stretch', type="primary"):
            st.session_state.show_ai_chat = True
            st.rerun()


def render_chat_modal(insight_engine, context: Dict, 
                     branch_id: str = None, category: str = None):
    """Render chat in a modal-like expander"""
    
    if not st.session_state.get('show_ai_chat', False):
        return
    
    with st.expander("AI Insights Assistant", expanded=True):
        if st.button("Close", key="close_chat_modal"):
            st.session_state.show_ai_chat = False
            st.rerun()
        
        render_chat_interface(insight_engine, context, branch_id, category, "modal")


# Suggested questions based on context
def get_contextual_suggestions(context: Dict, branch_id: str = None, 
                              category: str = None) -> List[str]:
    """Generate contextual suggested questions"""
    suggestions = []
    
    if branch_id:
        suggestions = [
            f"What's the risk profile for {branch_id}?",
            f"Why is {branch_id} trending {'up' if 'worsening' in str(context).lower() else 'down'}?",
            f"Which departments in {branch_id} need attention?",
            f"What's the 3-month forecast for {branch_id}?",
            f"Show me convergence signals for {branch_id}"
        ]
    elif category:
        suggestions = [
            f"What's driving {category} trends?",
            f"Which branches have highest {category}?",
            f"{category} vs industry benchmarks",
            f"Forecast for {category} next quarter",
            f"Top {category} subtypes"
        ]
    else:
        exec_summary = context.get('executive_summary', {})
        alerts = exec_summary.get('predictive_alerts', [])
        
        suggestions = [
            "What's the overall risk score?",
            "Which branches need immediate attention?",
            "Show me converging risk signals",
            "What are the top predictive alerts?",
            "Forecast for next 3 months"
        ]
        
        # Add alert-based suggestions
        for alert in alerts[:2]:
            if 'attrition' in alert.lower():
                suggestions.append("Who are the top attrition risks?")
            elif 'escalat' in alert.lower():
                suggestions.append("Which grievances might escalate?")
            elif 'hotspot' in alert.lower():
                suggestions.append("Show me the hotspot clusters")
    
    return suggestions[:5]


def render_suggestion_chips(suggestions: List[str], key_prefix: str, 
                           on_click_callback=None):
    """Render clickable suggestion chips"""
    st.markdown("**Suggested:**")
    cols = st.columns(min(len(suggestions), 3))
    
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"{key_prefix}_sugg_{i}", 
                        width='stretch', help="Click to ask"):
                if on_click_callback:
                    on_click_callback(suggestion)
                else:
                    # Store in session state for next run
                    st.session_state[f"{key_prefix}_pending_question"] = suggestion
                st.rerun()
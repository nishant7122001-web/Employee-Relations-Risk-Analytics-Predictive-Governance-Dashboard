"""
Reusable Filter Components for Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from src.utils import get_filter_options, apply_filters, init_session_state


def _seed_multiselect(key: str, default: List[Any], valid: Optional[List[Any]] = None):
    """Seed a multiselect widget's key in session state exactly once.

    Widgets below are created with a ``key`` but *without* a ``default``. The
    initial value is written to ``st.session_state`` here instead. This avoids
    Streamlit's "created with a default value but also had its value set via the
    Session State API" warning (which fired for ``global_date_range`` and was
    latent for every other keyed widget). When ``valid`` is provided, any stale
    selection not present in the current options is pruned so Streamlit does not
    raise on invalid values.
    """
    if key not in st.session_state:
        st.session_state[key] = list(default)
    if valid is not None:
        st.session_state[key] = [v for v in st.session_state[key] if v in valid]


def _seed_value(key: str, default: Any):
    """Seed a single-value widget's key in session state exactly once."""
    if key not in st.session_state:
        st.session_state[key] = default


def render_global_filters(data_dict: Dict[str, pd.DataFrame], 
                         key_prefix: str = "global",
                         show_advanced: bool = True) -> Dict[str, Any]:
    """Render global filter sidebar and return filter dict"""
    
    init_session_state()
    options = get_filter_options(data_dict)
    
    filters = st.session_state.filters.copy()
    
    with st.sidebar:
        st.markdown("### Global Filters")
        
        # Region filter
        if options['regions']:
            _seed_multiselect(
                f"{key_prefix}_regions",
                filters.get('regions', []),
                options['regions']
            )
            selected_regions = st.multiselect(
                "Region",
                options['regions'],
                key=f"{key_prefix}_regions"
            )
            filters['regions'] = selected_regions
            
            # Filter branches by region
            branches_df = data_dict['branches']
            if selected_regions:
                branch_options = branches_df[branches_df['region'].isin(selected_regions)]['branch_name'].tolist()
            else:
                branch_options = options['branches']
        else:
            branch_options = options['branches']
        
        # Branch filter
        if branch_options:
            _seed_multiselect(
                f"{key_prefix}_branches",
                [b for b in filters.get('branch_names', []) if b in branch_options],
                branch_options
            )
            selected_branches = st.multiselect(
                "Branch",
                branch_options,
                key=f"{key_prefix}_branches"
            )
            filters['branch_names'] = selected_branches
            
            # Map branch names to IDs
            branches_df = data_dict['branches']
            if selected_branches:
                branch_ids = branches_df[branches_df['branch_name'].isin(selected_branches)]['branch_id'].tolist()
                filters['branch_ids'] = branch_ids
            else:
                filters['branch_ids'] = []
        
        # Category filter
        if options['categories']:
            _seed_multiselect(
                f"{key_prefix}_categories",
                filters.get('categories', []),
                options['categories']
            )
            selected_cats = st.multiselect(
                "ER Category",
                options['categories'],
                key=f"{key_prefix}_categories"
            )
            filters['categories'] = selected_cats
        
        # Subtype filter (dependent on category)
        if options['subtypes']:
            subtypes_df = data_dict['events']
            if filters.get('categories'):
                subtypes_df = subtypes_df[subtypes_df['category'].isin(filters['categories'])]
            available_subtypes = sorted(subtypes_df['subtype'].unique().tolist())
            
            _seed_multiselect(
                f"{key_prefix}_subtypes",
                [s for s in filters.get('subtypes', []) if s in available_subtypes],
                available_subtypes
            )
            selected_subtypes = st.multiselect(
                "Subtype",
                available_subtypes,
                key=f"{key_prefix}_subtypes"
            )
            filters['subtypes'] = selected_subtypes
        
        # Date range filter
        events_df = data_dict['events']
        if 'event_date' in events_df.columns and len(events_df) > 0:
            min_date = events_df['event_date'].min().date()
            max_date = events_df['event_date'].max().date()
            
            date_key = f"{key_prefix}_date_range"
            existing_range = filters.get('date_range')
            if not (isinstance(existing_range, tuple) and len(existing_range) == 2):
                existing_range = (min_date, max_date)
            _seed_value(date_key, existing_range)
            
            date_range = st.date_input(
                "Date Range",
                min_value=min_date,
                max_value=max_date,
                key=date_key
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                filters['date_range'] = date_range
            else:
                filters['date_range'] = (min_date, max_date)
        
        # Advanced filters
        if show_advanced:
            with st.expander("Advanced Filters"):
                # Severity range
                _seed_value(f"{key_prefix}_severity", filters.get('severity_range', (1, 5)))
                severity_range = st.slider(
                    "Severity Range",
                    min_value=1, max_value=5,
                    key=f"{key_prefix}_severity"
                )
                filters['severity_range'] = severity_range
                
                # Escalated only
                _seed_value(f"{key_prefix}_escalated", filters.get('escalated_only', False))
                escalated_only = st.checkbox(
                    "Escalated Only",
                    key=f"{key_prefix}_escalated"
                )
                filters['escalated_only'] = escalated_only
                
                # Resolved only
                _seed_value(f"{key_prefix}_resolved", filters.get('resolved_only', False))
                resolved_only = st.checkbox(
                    "Resolved Only",
                    key=f"{key_prefix}_resolved"
                )
                filters['resolved_only'] = resolved_only
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset", key=f"{key_prefix}_reset", width='stretch'):
                from src.utils import reset_filters
                reset_filters()
                st.rerun()
        
        with col2:
            if st.button("Apply", key=f"{key_prefix}_apply", width='stretch', type="primary"):
                st.session_state.filters = filters
                st.success("Filters applied!")
                st.rerun()
        
        # Update session state
        st.session_state.filters = filters
    
    return filters


def render_branch_selector(branches_df: pd.DataFrame, 
                          key: str = "branch_selector",
                          label: str = "Select Branch") -> Optional[str]:
    """Render branch selector dropdown"""
    if len(branches_df) == 0:
        return None
    
    # Create display names
    branch_options = {}
    for _, row in branches_df.iterrows():
        display = f"{row['branch_name']} ({row['city']}, {row['region']})"
        branch_options[display] = row['branch_id']
    
    selected_display = st.selectbox(
        label,
        options=list(branch_options.keys()),
        key=key
    )
    
    return branch_options.get(selected_display)


def render_category_tabs(categories: List[str], 
                        key: str = "category_tabs",
                        default: str = None) -> str:
    """Render category tabs"""
    if default is None and categories:
        default = categories[0]
    
    tabs = st.tabs([f"{c}" for c in categories])
    return tabs


def render_time_granularity(key: str = "time_granularity") -> str:
    """Render time granularity selector"""
    return st.radio(
        "Time Granularity",
        ["Monthly", "Quarterly", "Yearly"],
        index=0,
        horizontal=True,
        key=key
    )


def render_risk_threshold_slider(key: str = "risk_threshold") -> float:
    """Render risk threshold slider"""
    return st.slider(
        "Risk Score Threshold",
        min_value=0.0, max_value=10.0,
        value=5.0, step=0.5,
        key=key,
        help="Filter branches by minimum risk score"
    )


def render_comparison_selector(branches_df: pd.DataFrame,
                              key: str = "comparison",
                              label: str = "Compare with") -> List[str]:
    """Render branch comparison selector"""
    
    branch_options = {}
    for _, row in branches_df.iterrows():
        display = f"{row['branch_name']} ({row['city']})"
        branch_options[display] = row['branch_id']
    
    selected = st.multiselect(
        label,
        options=list(branch_options.keys()),
        key=key
    )
    
    return [branch_options[s] for s in selected]


def get_filtered_data(data_dict: Dict[str, pd.DataFrame], 
                     filters: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Apply filters to all relevant dataframes"""
    filtered = {}
    
    # Filter events
    if 'events' in data_dict:
        filtered['events'] = apply_filters(data_dict['events'], filters)
    
    # Filter cases
    if 'cases' in data_dict:
        cases = data_dict['cases'].copy()
        if filters.get('branch_ids'):
            cases = cases[cases['branch_id'].isin(filters['branch_ids'])]
        if filters.get('categories'):
            cases = cases[cases['category'].isin(filters['categories'])]
        filtered['cases'] = cases
    
    # Filter kpis
    if 'kpis' in data_dict:
        kpis = data_dict['kpis'].copy()
        if filters.get('branch_ids'):
            kpis = kpis[kpis['branch_id'].isin(filters['branch_ids'])]
        if filters.get('date_range'):
            # KPIs have month/year - filter on the full year-month period
            start, end = filters['date_range']
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            if 'year' in kpis.columns:
                start_key = start.year * 100 + start.month
                end_key = end.year * 100 + end.month
                period_key = kpis['year'] * 100 + kpis['month']
                kpis = kpis[(period_key >= start_key) & (period_key <= end_key)]
            else:
                kpis = kpis[(kpis['month'] >= start.month) & (kpis['month'] <= end.month)]
        filtered['kpis'] = kpis
    
    # Branches and employees - filter by branch_ids
    if 'branches' in data_dict and filters.get('branch_ids'):
        filtered['branches'] = data_dict['branches'][
            data_dict['branches']['branch_id'].isin(filters['branch_ids'])
        ]
    else:
        filtered['branches'] = data_dict.get('branches', pd.DataFrame())
    
    if 'employees' in data_dict and filters.get('branch_ids'):
        filtered['employees'] = data_dict['employees'][
            data_dict['employees']['branch_id'].isin(filters['branch_ids'])
        ]
    else:
        filtered['employees'] = data_dict.get('employees', pd.DataFrame())
    
    return filtered


def render_filter_summary(filters: Dict[str, Any]):
    """Render active filter summary"""
    active = []
    
    if filters.get('branch_ids'):
        active.append(f"{len(filters['branch_ids'])} branch(es)")
    
    if filters.get('categories'):
        active.append(f"{', '.join(filters['categories'])}")
    
    if filters.get('subtypes'):
        active.append(f"{len(filters['subtypes'])} subtype(s)")
    
    if filters.get('date_range'):
        start, end = filters['date_range']
        active.append(f"{start} to {end}")
    
    if filters.get('escalated_only'):
        active.append("Escalated only")
    
    if filters.get('resolved_only'):
        active.append("Resolved only")
    
    if active:
        st.markdown("**Active Filters:** " + " | ".join(active))
    else:
        st.markdown("*No active filters*")
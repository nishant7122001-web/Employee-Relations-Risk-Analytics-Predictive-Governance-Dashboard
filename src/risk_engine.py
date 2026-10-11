"""
Employee Relations Risk Engine
Core analytics: risk scoring, clustering, predictive models, trend detection
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import hashlib
import warnings
warnings.filterwarnings('ignore')


def _stable_fraction(key) -> float:
    """Deterministic pseudo-random fraction in [0, 1) derived from a key."""
    digest = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
    return (int(digest[:8], 16) % 10000) / 10000.0

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ERRiskEngine:
    """Main risk analytics engine for ER data"""
    
    def __init__(self, events_df, kpis_df, branches_df, employees_df, cases_df=None):
        self.events_df = events_df
        self.kpis_df = kpis_df
        self.branches_df = branches_df
        self.employees_df = employees_df
        self.cases_df = cases_df
        
        self.risk_weights = {
            'grievance': 0.25,
            'disciplinary': 0.20,
            'absenteeism': 0.15,
            'attrition': 0.20,
            'whistleblower': 0.10,
            'escalation': 0.10
        }
        
        self.models = {}
        self.scalers = {}
        self.cluster_labels = None

        # Per-instance memoization cache for expensive, data-static
        # computations (Prophet forecasts, model inference, org/branch
        # insight aggregation). The engine is stored via
        # ``@st.cache_resource`` in ``dashboard/shared_init.py`` so this
        # cache survives across Streamlit reruns and browser sessions for the
        # life of the server process -- computing each item at most once.
        self._cache = {}
        for _name in (
            'predict_attrition_risk',
            'predict_escalation_risk',
            'forecast_branch_category',
            'forecast_org_risk',
            'generate_org_insights',
            'generate_branch_insights',
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
                # Unhashable arguments: skip caching rather than fail.
                return func(*args, **kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        wrapper.__name__ = name
        wrapper.__doc__ = getattr(func, '__doc__', None)
        return wrapper

    # ============================================================
    # RISK SCORING
    # ============================================================
    def compute_branch_risk_score(self, kpi_row):
        """Compute composite risk score from KPI components"""
        components = {
            'grievance': min(kpi_row.get('grievance_rate', 0) * 2, 10),
            'disciplinary': min(kpi_row.get('disciplinary_rate', 0) * 3, 10),
            'absenteeism': min(kpi_row.get('absenteeism_rate', 0) * 0.5, 10),
            'attrition': min(kpi_row.get('attrition_rate_annualized', 0) * 0.8, 10),
            'whistleblower': min(kpi_row.get('whistleblower_rate', 0) * 20, 10),
            'escalation': min(kpi_row.get('grievance_escalation_rate', 0), 10)
        }
        
        score = sum(components[k] * self.risk_weights[k] for k in components)
        return min(max(score, 0), 10)
    
    def compute_convergence_index(self, branch_kpis):
        """Detect converging risk signals (multiple categories rising together)"""
        # Get last 3 months for trend
        recent = branch_kpis.tail(3)
        if len(recent) < 2:
            return 0.0
        
        # Categories to check for convergence
        risk_cols = ['grievance_rate', 'disciplinary_rate', 'absenteeism_rate', 
                    'attrition_rate_annualized', 'whistleblower_rate']
        
        # Count how many categories are trending up
        trending_up = 0
        for col in risk_cols:
            if col in recent.columns:
                trend = recent[col].iloc[-1] - recent[col].iloc[0]
                if trend > 0:
                    trending_up += 1
        
        # Convergence index: proportion of categories trending up, weighted by magnitude
        magnitudes = []
        for col in risk_cols:
            if col in recent.columns:
                trend = recent[col].iloc[-1] - recent[col].iloc[0]
                if trend > 0:
                    magnitudes.append(trend / (recent[col].iloc[0] + 0.01))
        
        avg_magnitude = np.mean(magnitudes) if magnitudes else 0
        convergence = (trending_up / len(risk_cols)) * min(avg_magnitude, 2)
        
        return min(convergence, 1.0)
    
    def compute_risk_trend(self, branch_kpis, months=6):
        """Compute risk score trend over specified months"""
        recent = branch_kpis.tail(months)
        if len(recent) < 2:
            return 0.0, 'Stable'
        
        # Compute risk scores for each month
        recent = recent.copy()
        recent['risk_score'] = recent.apply(self.compute_branch_risk_score, axis=1)
        
        trend = recent['risk_score'].iloc[-1] - recent['risk_score'].iloc[0]
        slope = np.polyfit(range(len(recent)), recent['risk_score'], 1)[0]
        
        if slope > 0.1:
            direction = 'Worsening'
        elif slope < -0.1:
            direction = 'Improving'
        else:
            direction = 'Stable'
        
        return round(slope, 3), direction
    
    # ============================================================
    # CLUSTERING / HOTSPOT DETECTION
    # ============================================================
    def detect_hotspots(self, min_samples=3, eps=1.5):
        """DBSCAN clustering on branch risk profiles to find hotspots"""
        # Prepare features for clustering
        latest_kpis = self.kpis_df.groupby('branch_id').last().reset_index()
        
        features = ['grievance_rate', 'disciplinary_rate', 'absenteeism_rate',
                   'attrition_rate_annualized', 'whistleblower_rate', 
                   'grievance_escalation_rate', 'risk_score']
        
        # Ensure all features exist
        for f in features:
            if f not in latest_kpis.columns:
                latest_kpis[f] = 0
        
        X = latest_kpis[features].fillna(0)
        self.scalers['hotspot'] = StandardScaler()
        X_scaled = self.scalers['hotspot'].fit_transform(X)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        latest_kpis['cluster'] = labels
        latest_kpis['is_hotspot'] = labels != -1
        
        # Merge with branch info
        result = latest_kpis.merge(self.branches_df[['branch_id', 'branch_name', 'city', 'region']], 
                                   on='branch_id', how='left')
        
        self.cluster_labels = result
        return result
    
    def get_hotspot_summary(self):
        """Get summary of detected hotspots"""
        if self.cluster_labels is None:
            self.detect_hotspots()
        
        hotspots = self.cluster_labels[self.cluster_labels['is_hotspot']]
        if len(hotspots) == 0:
            return pd.DataFrame()
        
        summary = hotspots.groupby('cluster').agg(
            branch_count=('branch_id', 'count'),
            avg_risk_score=('risk_score', 'mean'),
            regions=('region', lambda x: ', '.join(x.unique())),
            top_branches=('branch_name', lambda x: ', '.join(x.head(3)))
        ).reset_index()
        
        summary = summary.sort_values('avg_risk_score', ascending=False)
        return summary
    
    # ============================================================
    # PREDICTIVE MODELS
    # ============================================================
    def prepare_attrition_features(self):
        """Prepare features for attrition risk model"""
        # Employee-level features
        emp_features = self.employees_df.copy()
        
        # Add recent ER event counts per employee
        recent_events = self.events_df[self.events_df['event_date'] >= datetime(2023, 10, 1)]
        emp_event_counts = recent_events.groupby(['employee_id', 'category']).size().unstack(fill_value=0)
        emp_event_counts.columns = [f'events_{c.lower()}' for c in emp_event_counts.columns]
        emp_features = emp_features.merge(emp_event_counts, on='employee_id', how='left')
        emp_features = emp_features.fillna(0)
        
        # Target: attrition in last quarter
        attrition_events = self.events_df[
            (self.events_df['category'] == 'Attrition') & 
            (self.events_df['event_date'] >= datetime(2023, 10, 1))
        ]
        attrition_emps = set(attrition_events['employee_id'].unique())
        emp_features['attrition_target'] = emp_features['employee_id'].isin(attrition_emps).astype(int)
        
        # Feature columns
        feature_cols = ['tenure_years', 'age', 'performance_rating', 'flight_risk_base',
                       'high_performer'] + [c for c in emp_features.columns if c.startswith('events_')]
        
        # Add branch risk
        branch_risk = self.kpis_df.groupby('branch_id')['risk_score'].mean().reset_index()
        branch_risk.columns = ['branch_id', 'branch_avg_risk']
        emp_features = emp_features.merge(branch_risk, on='branch_id', how='left')
        feature_cols.append('branch_avg_risk')
        
        return emp_features, feature_cols
    
    def train_attrition_model(self):
        """Train attrition risk prediction model"""
        emp_features, feature_cols = self.prepare_attrition_features()
        
        X = emp_features[feature_cols].fillna(0)
        y = emp_features['attrition_target']
        
        if y.sum() < 10:
            # Not enough positive samples, return dummy model
            self.models['attrition'] = None
            return {'auc': 0.5, 'n_samples': len(X), 'n_positive': int(y.sum())}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        if XGB_AVAILABLE:
            model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric='auc'
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        
        model.fit(X_train, y_train)
        self.models['attrition'] = model
        self.scalers['attrition'] = None  # Tree models don't need scaling
        
        # Evaluate
        from sklearn.metrics import roc_auc_score
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        return {'auc': round(auc, 3), 'n_samples': len(X), 'n_positive': int(y.sum())}
    
    def predict_attrition_risk(self, top_n=50):
        """Predict attrition risk for all employees"""
        if 'attrition' not in self.models or self.models['attrition'] is None:
            # Return heuristic-based ranking
            emp_features, _ = self.prepare_attrition_features()
            emp_features['risk_score'] = (
                emp_features['flight_risk_base'] * 0.4 +
                (1 - emp_features['performance_rating'] / 5) * 0.2 +
                emp_features.get('branch_avg_risk', 3) / 10 * 0.2 +
                emp_features.get('events_grievance', 0) * 0.1 +
                emp_features.get('events_disciplinary', 0) * 0.1
            )
            return emp_features.nlargest(top_n, 'risk_score')[
                ['employee_id', 'branch_id', 'department', 'grade', 
                 'performance_rating', 'tenure_years', 'risk_score']
            ]
        
        emp_features, feature_cols = self.prepare_attrition_features()
        X = emp_features[feature_cols].fillna(0)
        
        probs = self.models['attrition'].predict_proba(X)[:, 1]
        emp_features['attrition_risk'] = probs
        
        return emp_features.nlargest(top_n, 'attrition_risk')[
            ['employee_id', 'branch_id', 'department', 'grade',
             'performance_rating', 'tenure_years', 'attrition_risk']
        ]
    
    def train_escalation_model(self):
        """Train grievance escalation prediction model"""
        grievances = self.events_df[self.events_df['category'] == 'Grievance'].copy()
        
        if len(grievances) < 50:
            self.models['escalation'] = None
            return {'auc': 0.5, 'n_samples': len(grievances)}
        
        # Features
        grievances['month'] = grievances['event_date'].dt.month
        grievances['severity_sq'] = grievances['severity'] ** 2
        
        # Add branch context
        branch_context = self.kpis_df.groupby('branch_id').agg({
            'grievance_rate': 'mean',
            'grievance_escalation_rate': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        branch_context.columns = ['branch_id', 'branch_grievance_rate', 'branch_escalation_rate', 'branch_risk']
        
        grievances = grievances.merge(branch_context, on='branch_id', how='left')
        grievances = grievances.fillna(0)
        
        feature_cols = ['severity', 'severity_sq', 'month', 'branch_grievance_rate', 
                       'branch_escalation_rate', 'branch_risk']
        
        # Add employee features
        emp_features = self.employees_df[['employee_id', 'tenure_years', 'grade', 'performance_rating']].copy()
        emp_features['grade_ordinal'] = emp_features['grade'].map({f'Grade {i}': i for i in range(1, 8)})
        grievances = grievances.merge(emp_features, on='employee_id', how='left')
        feature_cols += ['tenure_years', 'grade_ordinal', 'performance_rating']
        
        X = grievances[feature_cols].fillna(0)
        y = grievances['escalated'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        if XGB_AVAILABLE:
            model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        
        model.fit(X_train, y_train)
        self.models['escalation'] = model
        
        from sklearn.metrics import roc_auc_score
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        return {'auc': round(auc, 3), 'n_samples': len(X), 'n_positive': int(y.sum())}
    
    def predict_escalation_risk(self, grievance_events=None):
        """Predict escalation risk for grievance cases"""
        if grievance_events is None:
            grievance_events = self.events_df[self.events_df['category'] == 'Grievance'].copy()
        
        if 'escalation' not in self.models or self.models['escalation'] is None:
            # Heuristic (deterministic per event)
            jitter = grievance_events['event_id'].map(lambda eid: _stable_fraction(eid))
            grievance_events['escalation_risk'] = (
                grievance_events['severity'] / 5 * 0.5 + jitter * 0.5
            )
            return grievance_events.nlargest(20, 'escalation_risk')[
                ['event_id', 'branch_id', 'employee_id', 'severity', 'subtype', 'escalation_risk']
            ]
        
        # Prepare features same as training
        grievance_events = grievance_events.copy()
        grievance_events['month'] = grievance_events['event_date'].dt.month
        grievance_events['severity_sq'] = grievance_events['severity'] ** 2
        
        branch_context = self.kpis_df.groupby('branch_id').agg({
            'grievance_rate': 'mean', 'grievance_escalation_rate': 'mean', 'risk_score': 'mean'
        }).reset_index()
        branch_context.columns = ['branch_id', 'branch_grievance_rate', 'branch_escalation_rate', 'branch_risk']
        grievance_events = grievance_events.merge(branch_context, on='branch_id', how='left')
        
        emp_features = self.employees_df[['employee_id', 'tenure_years', 'grade', 'performance_rating']].copy()
        emp_features['grade_ordinal'] = emp_features['grade'].map({f'Grade {i}': i for i in range(1, 8)})
        grievance_events = grievance_events.merge(emp_features, on='employee_id', how='left')
        grievance_events = grievance_events.fillna(0)
        
        feature_cols = ['severity', 'severity_sq', 'month', 'branch_grievance_rate', 
                       'branch_escalation_rate', 'branch_risk', 'tenure_years', 'grade_ordinal', 'performance_rating']
        
        X = grievance_events[feature_cols].fillna(0)
        probs = self.models['escalation'].predict_proba(X)[:, 1]
        grievance_events['escalation_risk'] = probs
        
        return grievance_events.nlargest(20, 'escalation_risk')[
            ['event_id', 'branch_id', 'employee_id', 'severity', 'subtype', 'escalation_risk']
        ]
    
    # ============================================================
    # TREND FORECASTING
    # ============================================================
    def _last_period(self):
        """Return (year, month) of the latest available KPI period."""
        try:
            if 'year' in self.kpis_df.columns and 'month' in self.kpis_df.columns and len(self.kpis_df) > 0:
                ordered = self.kpis_df.copy()
                ordered['_key'] = ordered['year'] * 100 + ordered['month']
                last = ordered.loc[ordered['_key'].idxmax()]
                return int(last['year']), int(last['month'])
        except Exception:
            pass
        m = int(self.kpis_df['month'].max()) if len(self.kpis_df) > 0 else 12
        return 2023, m

    @staticmethod
    def _next_month_labels(year, month, periods):
        """Generate the next `periods` YYYY-MM labels after (year, month)."""
        labels = []
        y, m = int(year), int(month)
        for _ in range(periods):
            m += 1
            if m > 12:
                m = 1
                y += 1
            labels.append(f"{y}-{m:02d}")
        return labels

    def _sorted_branch_kpis(self, branch_id):
        branch_kpis = self.kpis_df[self.kpis_df['branch_id'] == branch_id].copy()
        if 'year' in branch_kpis.columns and 'month' in branch_kpis.columns:
            branch_kpis = branch_kpis.sort_values(['year', 'month'])
        else:
            branch_kpis = branch_kpis.sort_values('month')
        return branch_kpis

    def forecast_branch_category(self, branch_id, category, periods=3):
        """Forecast category trend for a branch using Prophet"""
        if not PROPHET_AVAILABLE:
            return self._simple_forecast(branch_id, category, periods)
        
        # Get monthly data
        branch_kpis = self._sorted_branch_kpis(branch_id)
        
        if len(branch_kpis) < 6:
            return self._simple_forecast(branch_id, category, periods)
        
        # Map category to KPI column
        col_map = {
            'Grievance': 'grievance_rate',
            'Disciplinary': 'disciplinary_rate',
            'Absenteeism': 'absenteeism_rate',
            'Attrition': 'attrition_rate_annualized',
            'Whistleblower': 'whistleblower_rate'
        }
        col = col_map.get(category, 'grievance_rate')
        
        if col not in branch_kpis.columns:
            return self._simple_forecast(branch_id, category, periods)
        
        # Prepare for Prophet
        df = branch_kpis[[col]].copy()
        if 'year' in branch_kpis.columns:
            df['ds'] = pd.to_datetime(dict(
                year=branch_kpis['year'], month=branch_kpis['month'], day=1
            ))
        else:
            df['ds'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(
                (branch_kpis['month'] - 1) * 30, unit='D'
            )
        df['y'] = branch_kpis[col]
        df = df[['ds', 'y']].dropna()
        
        if len(df) < 6:
            return self._simple_forecast(branch_id, category, periods)
        
        try:
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.1,
                interval_width=0.8
            )
            model.fit(df)
            
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)
            
            # Return last 'periods' predictions
            result = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            result['ds'] = result['ds'].dt.strftime('%Y-%m')
            result.columns = ['month', 'forecast', 'lower', 'upper']
            return result
        except Exception:
            return self._simple_forecast(branch_id, category, periods)
    
    def _simple_forecast(self, branch_id, category, periods=3):
        """Simple linear trend fallback"""
        branch_kpis = self._sorted_branch_kpis(branch_id)
        
        col_map = {
            'Grievance': 'grievance_rate',
            'Disciplinary': 'disciplinary_rate',
            'Absenteeism': 'absenteeism_rate',
            'Attrition': 'attrition_rate_annualized',
            'Whistleblower': 'whistleblower_rate'
        }
        col = col_map.get(category, 'grievance_rate')
        
        year, month = self._last_period()
        labels = self._next_month_labels(year, month, periods)
        
        if col not in branch_kpis.columns or len(branch_kpis) < 3:
            last_val = float(branch_kpis[col].iloc[-1]) if (col in branch_kpis.columns and len(branch_kpis) > 0) else 0.0
            return pd.DataFrame({
                'month': labels,
                'forecast': [last_val] * periods,
                'lower': [last_val * 0.8] * periods,
                'upper': [last_val * 1.2] * periods
            })
        
        y = branch_kpis[col].values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        
        forecasts = []
        for i in range(1, periods+1):
            next_val = intercept + slope * (len(y) + i)
            forecasts.append(max(next_val, 0))
        
        return pd.DataFrame({
            'month': labels,
            'forecast': forecasts,
            'lower': [f * 0.8 for f in forecasts],
            'upper': [f * 1.2 for f in forecasts]
        })
    
    def forecast_org_risk(self, periods=3):
        """Forecast organization-wide risk score"""
        df_src = self.kpis_df.copy()
        group_cols = ['year', 'month'] if 'year' in df_src.columns else ['month']
        org_kpis = df_src.groupby(group_cols).agg({
            'grievance_rate': 'mean',
            'disciplinary_rate': 'mean',
            'absenteeism_rate': 'mean',
            'attrition_rate_annualized': 'mean',
            'whistleblower_rate': 'mean',
            'grievance_escalation_rate': 'mean'
        }).reset_index()
        
        if 'year' in org_kpis.columns:
            org_kpis = org_kpis.sort_values(['year', 'month'])
        else:
            org_kpis = org_kpis.sort_values('month')
        
        org_kpis['risk_score'] = org_kpis.apply(self.compute_branch_risk_score, axis=1)
        
        year, month = self._last_period()
        labels = self._next_month_labels(year, month, periods)
        
        def _simple():
            y = org_kpis['risk_score'].values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            forecasts = [max(intercept + slope * (len(y) + i), 0) for i in range(1, periods + 1)]
            return pd.DataFrame({
                'month': labels,
                'forecast': forecasts,
                'lower': [f * 0.9 for f in forecasts],
                'upper': [f * 1.1 for f in forecasts]
            })
        
        if not PROPHET_AVAILABLE or len(org_kpis) < 6:
            return _simple()
        
        df = pd.DataFrame({'risk_score': org_kpis['risk_score'].values})
        if 'year' in org_kpis.columns:
            df['ds'] = pd.to_datetime(dict(year=org_kpis['year'], month=org_kpis['month'], day=1))
        else:
            df['ds'] = pd.to_datetime('2023-01-01') + pd.to_timedelta((org_kpis['month'] - 1) * 30, unit='D')
        df['y'] = df['risk_score']
        df = df[['ds', 'y']].dropna()
        
        try:
            model = Prophet(yearly_seasonality=False, weekly_seasonality=False, 
                          daily_seasonality=False, interval_width=0.8)
            model.fit(df)
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)
            result = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            result['ds'] = result['ds'].dt.strftime('%Y-%m')
            result.columns = ['month', 'forecast', 'lower', 'upper']
            return result
        except Exception:
            return _simple()
    
    # ============================================================
    # WHAT-IF SIMULATOR
    # ============================================================
    def simulate_intervention(self, branch_id, intervention_type, intensity=0.5):
        """Simulate impact of intervention on branch risk"""
        branch_kpis = self.kpis_df[self.kpis_df['branch_id'] == branch_id].copy()
        if len(branch_kpis) == 0:
            return None
        
        latest = branch_kpis.iloc[-1].copy()
        current_risk = self.compute_branch_risk_score(latest)
        
        # Intervention effects (reduction factors)
        effects = {
            'grievance_mediation': {'grievance_rate': 0.3, 'escalation': 0.4},
            'manager_training': {'grievance_rate': 0.2, 'disciplinary_rate': 0.15, 'attrition_rate_annualized': 0.2},
            'wellbeing_program': {'absenteeism_rate': 0.25, 'attrition_rate_annualized': 0.15},
            'compensation_review': {'grievance_rate': 0.35, 'attrition_rate_annualized': 0.3},
            'policy_reinforcement': {'disciplinary_rate': 0.3, 'staff_accountability': 0.2},
            'whistleblower_protection': {'whistleblower_rate': -0.1, 'escalation': 0.2}  # May increase reporting
        }
        
        effect = effects.get(intervention_type, {})
        simulated = latest.copy()
        
        for k, reduction in effect.items():
            if k in simulated.index:
                simulated[k] = simulated[k] * (1 - reduction * intensity)
        
        new_risk = self.compute_branch_risk_score(simulated)
        risk_reduction = current_risk - new_risk
        
        return {
            'branch_id': branch_id,
            'intervention': intervention_type,
            'intensity': intensity,
            'current_risk': round(current_risk, 2),
            'projected_risk': round(new_risk, 2),
            'risk_reduction': round(risk_reduction, 2),
            'reduction_pct': round(risk_reduction / current_risk * 100, 1) if current_risk > 0 else 0
        }
    
    # ============================================================
    # COMPREHENSIVE INSIGHTS GENERATION
    # ============================================================
    def generate_org_insights(self):
        """Generate comprehensive organizational insights"""
        latest_kpis = self.kpis_df.groupby('branch_id').last().reset_index()
        latest_kpis['risk_score'] = latest_kpis.apply(self.compute_branch_risk_score, axis=1)
        latest_kpis['convergence'] = latest_kpis['branch_id'].apply(
            lambda b: self.compute_convergence_index(self.kpis_df[self.kpis_df['branch_id'] == b])
        )
        
        # Merge branch info
        latest_kpis = latest_kpis.merge(
            self.branches_df[['branch_id', 'branch_name', 'city', 'region']], on='branch_id'
        )
        
        # Overall stats
        org_risk = latest_kpis['risk_score'].mean()
        high_risk_branches = latest_kpis[latest_kpis['risk_score'] >= 7]
        critical_branches = latest_kpis[latest_kpis['risk_score'] >= 8.5]
        
        # Top risk drivers
        driver_cols = ['grievance_rate', 'disciplinary_rate', 'absenteeism_rate', 
                      'attrition_rate_annualized', 'whistleblower_rate']
        driver_avgs = {c: latest_kpis[c].mean() for c in driver_cols if c in latest_kpis.columns}
        top_drivers = sorted(driver_avgs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Trending branches
        trends = []
        for _, row in latest_kpis.iterrows():
            branch_kpis = self.kpis_df[self.kpis_df['branch_id'] == row['branch_id']]
            slope, direction = self.compute_risk_trend(branch_kpis)
            if direction != 'Stable':
                trends.append({
                    'branch_id': row['branch_id'],
                    'branch_name': row['branch_name'],
                    'direction': direction,
                    'slope': slope
                })
        
        if trends:
            trends_df = pd.DataFrame(trends)
            worsening = trends_df[trends_df['direction'] == 'Worsening'].nlargest(5, 'slope')
            improving = trends_df[trends_df['direction'] == 'Improving'].nsmallest(5, 'slope')
        else:
            worsening = pd.DataFrame()
            improving = pd.DataFrame()
        
        # Category trends
        agg_cols = {
            'grievance_rate': 'mean', 'disciplinary_rate': 'mean',
            'absenteeism_rate': 'mean', 'attrition_rate_annualized': 'mean',
            'whistleblower_rate': 'mean', 'risk_score': 'mean'
        }
        group_cols = ['year', 'month'] if 'year' in self.kpis_df.columns else ['month']
        org_monthly = self.kpis_df.groupby(group_cols).agg(agg_cols).reset_index()
        
        category_trends = {}
        for cat in ['grievance_rate', 'disciplinary_rate', 'absenteeism_rate', 
                   'attrition_rate_annualized', 'whistleblower_rate']:
            if len(org_monthly) >= 3:
                recent = org_monthly[cat].tail(3)
                trend = recent.iloc[-1] - recent.iloc[0]
                category_trends[cat] = round(trend, 3)
        
        return {
            'org_risk_score': round(org_risk, 2),
            'org_risk_level': pd.cut([org_risk], bins=[0,3,5,7,10], labels=['Low','Medium','High','Critical'])[0],
            'total_branches': len(latest_kpis),
            'high_risk_count': len(high_risk_branches),
            'critical_count': len(critical_branches),
            'top_risk_drivers': top_drivers,
            'top_risk_branches': high_risk_branches.nlargest(5, 'risk_score')[
                ['branch_id', 'branch_name', 'city', 'region', 'risk_score', 'convergence']
            ].to_dict('records'),
            'worsening_branches': worsening.to_dict('records') if len(worsening) > 0 else [],
            'improving_branches': improving.to_dict('records') if len(improving) > 0 else [],
            'category_trends': category_trends,
            'convergence_branches': latest_kpis.nlargest(5, 'convergence')[
                ['branch_id', 'branch_name', 'city', 'convergence']
            ].to_dict('records')
        }
    
    def generate_branch_insights(self, branch_id):
        """Generate detailed insights for a specific branch"""
        branch_kpis = self.kpis_df[self.kpis_df['branch_id'] == branch_id].copy()
        branch_kpis = branch_kpis.sort_values('month')
        
        if len(branch_kpis) == 0:
            return {}
        
        latest = branch_kpis.iloc[-1]
        branch_info = self.branches_df[self.branches_df['branch_id'] == branch_id].iloc[0]
        
        risk_score = self.compute_branch_risk_score(latest)
        convergence = self.compute_convergence_index(branch_kpis)
        slope, trend_dir = self.compute_risk_trend(branch_kpis)
        
        # Category breakdown
        categories = {
            'Grievance': {
                'rate': latest.get('grievance_rate', 0),
                'trend': latest.get('grievance_rate', 0) - branch_kpis['grievance_rate'].iloc[0] if len(branch_kpis) > 1 else 0,
                'escalation_rate': latest.get('grievance_escalation_rate', 0),
                'avg_resolution': latest.get('grievance_avg_resolution', 0)
            },
            'Disciplinary': {
                'rate': latest.get('disciplinary_rate', 0),
                'trend': latest.get('disciplinary_rate', 0) - branch_kpis['disciplinary_rate'].iloc[0] if len(branch_kpis) > 1 else 0,
                'dismissal_rate': latest.get('dismissal_rate', 0)
            },
            'Absenteeism': {
                'rate': latest.get('absenteeism_rate', 0),
                'trend': latest.get('absenteeism_rate', 0) - branch_kpis['absenteeism_rate'].iloc[0] if len(branch_kpis) > 1 else 0,
                'bradford_factor': latest.get('bradford_factor', 0),
                'pattern_index': latest.get('absenteeism_pattern_index', 0)
            },
            'Attrition': {
                'rate': latest.get('attrition_rate_annualized', 0),
                'trend': latest.get('attrition_rate_annualized', 0) - branch_kpis['attrition_rate_annualized'].iloc[0] if len(branch_kpis) > 1 else 0,
                'voluntary_pct': latest.get('voluntary_attrition_pct', 0),
                'regrettable_pct': latest.get('regrettable_attrition_pct', 0)
            },
            'Whistleblower': {
                'rate': latest.get('whistleblower_rate', 0),
                'trend': latest.get('whistleblower_rate', 0) - branch_kpis['whistleblower_rate'].iloc[0] if len(branch_kpis) > 1 else 0,
                'investigation_completion': latest.get('wb_investigation_completion', 0)
            }
        }
        
        # Department breakdown
        branch_emp = self.employees_df[self.employees_df['branch_id'] == branch_id]
        dept_breakdown = branch_emp.groupby('department').agg(
            headcount=('employee_id', 'count'),
            avg_performance=('performance_rating', 'mean'),
            avg_tenure=('tenure_years', 'mean'),
            high_performer_pct=('high_performer', 'mean')
        ).round(2).reset_index()
        
        # Recent events
        recent_events = self.events_df[
            (self.events_df['branch_id'] == branch_id) & 
            (self.events_df['event_date'] >= datetime(2023, 10, 1))
        ]
        event_summary = recent_events.groupby('category').agg(
            count=('event_id', 'count'),
            avg_severity=('severity', 'mean'),
            escalated_pct=('escalated', 'mean')
        ).round(2).reset_index()
        
        # Forecasts
        forecasts = {}
        for cat in ['Grievance', 'Disciplinary', 'Absenteeism', 'Attrition', 'Whistleblower']:
            forecasts[cat] = self.forecast_branch_category(branch_id, cat, 3)
        
        org_forecast = self.forecast_org_risk(3)
        
        return {
            'branch_id': branch_id,
            'branch_name': branch_info['branch_name'],
            'city': branch_info['city'],
            'region': branch_info['region'],
            'archetype': branch_info['archetype'],
            'headcount': int(latest['headcount']),
            'risk_score': round(risk_score, 2),
            'risk_level': pd.cut([risk_score], bins=[0,3,5,7,10], labels=['Low','Medium','High','Critical'])[0],
            'convergence_index': round(convergence, 3),
            'trend_direction': trend_dir,
            'trend_slope': round(slope, 3),
            'categories': categories,
            'department_breakdown': dept_breakdown.to_dict('records'),
            'recent_events': event_summary.to_dict('records'),
            'forecasts': forecasts,
            'org_forecast': org_forecast.to_dict('records')
        }


def create_risk_engine(events_df, kpis_df, branches_df, employees_df, cases_df=None):
    """Factory function to create and initialize risk engine"""
    engine = ERRiskEngine(events_df, kpis_df, branches_df, employees_df, cases_df)
    
    # Train models
    print("Training attrition model...")
    engine.train_attrition_model()
    
    print("Training escalation model...")
    engine.train_escalation_model()
    
    print("Detecting hotspots...")
    engine.detect_hotspots()
    
    return engine
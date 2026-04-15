import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import ast
import datetime
import random
import json

# --- HELPER: SAFE PARSER ---
def safe_parse_data(data_str: str):
    """Safely converts string data to list, handling empty/error messages."""
    clean_str = data_str.strip()
    if not clean_str.startswith("["):
        return None, f"Data Retrieval Error: {clean_str}"
    try:
        data = ast.literal_eval(clean_str)
        if not data: return None, "Error: Dataset is empty."
        return data, None
    except Exception as e:
        return None, f"Parsing Error: {str(e)}"

# --- 1. FORECASTING ENGINE (ENHANCED WITH CONFIDENCE INTERVALS) ---
def perform_forecast(data_str: str):
    """Enhanced forecasting with confidence intervals and trend analysis"""
    try:
        data, error = safe_parse_data(data_str)
        if error: return error
        
        df = pd.DataFrame(data)
        cols = df.columns.tolist()
        if len(cols) < 2: return "Error: Not enough columns to forecast."

        # Parse dates
        try:
            if len(cols) >= 3:
                # Handle Year, Month, Value format
                df['Date'] = pd.to_datetime(dict(year=df[cols[0]], month=df[cols[1]], day=1), errors='coerce')
                y_col = cols[2]
            else:
                # Handle Date, Value format
                df['Date'] = pd.to_datetime(df[cols[0]], errors='coerce')
                y_col = cols[1]
        except Exception as e:
            logger.error(f"Date parsing failed: {e}")
            return "Error: Could not parse date column."

        # Drop rows with invalid dates (if any)
        df = df.dropna(subset=['Date'])
        
        if len(df) < 2:
            return "Error: Not enough valid historical data points for forecasting (minimum 2 required)."

        df = df.sort_values('Date')
        df['Time_Ordinal'] = df['Date'].apply(lambda date: date.toordinal())
        
        X = df[['Time_Ordinal']]
        y = df[y_col]
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate trend strength
        slope = model.coef_[0]
        mean_y = y.mean()
        trend_strength = abs(slope / mean_y) * 100 if mean_y != 0 else 0
        
        # Determine trend direction
        if trend_strength < 5:
            trend = "Stable"
            trend_emoji = "➡️"
        elif slope > 0:
            trend = "Growing"
            trend_emoji = "📈"
        else:
            trend = "Declining"
            trend_emoji = "📉"
        
        # Calculate R² score
        predictions = model.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - mean_y) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Forecast next period
        last_date = df['Date'].iloc[-1]
        next_date = last_date + datetime.timedelta(days=30)
        next_ordinal = np.array([[next_date.toordinal()]])
        
        prediction = model.predict(next_ordinal)[0]
        prediction = round(prediction, 1)
        
        # Calculate confidence interval (using residual standard error)
        residuals = y - predictions
        std_error = np.std(residuals)
        confidence_lower = round(prediction - 1.96 * std_error, 1)
        confidence_upper = round(prediction + 1.96 * std_error, 1)
        
        # Build response
        labels = df['Date'].dt.strftime('%Y-%m').tolist()
        values = df[y_col].tolist()
        labels.append(f"{next_date.strftime('%Y-%m')} (Forecast)")
        values.append(prediction)
        
        chart_json = {
            "chart_type": "line",
            "title": f"Forecast: {y_col} Trend Analysis",
            "x_axis": "Period",
            "y_axis": y_col,
            "data": [{"label": label, "value": value} for label, value in zip(labels, values)]
        }
        
        # ✅ EINSTEIN-STYLE INSIGHTS
        summary = f"""📊 **Predictive Forecast Analysis**

**Key Findings:**
- **Trend:** {trend} {trend_emoji} ({trend_strength:.1f}% change rate)
- **Forecast (Next Period):** **{prediction:,.1f}**
- **Confidence Range:** {confidence_lower:,.1f} to {confidence_upper:,.1f} (95% confidence)
- **Model Accuracy:** {r2_score*100:.1f}% variance explained

**Historical Context:**
Based on {len(df)} historical periods:
- **Average:** {mean_y:.1f}
- **Highest:** {y.max():.1f}
- **Lowest:** {y.min():.1f}

**📌 Recommendation:**
"""
        
        if trend == "Growing":
            summary += f"Your {y_col} is trending upward. Consider scaling operations to meet growing demand."
        elif trend == "Declining":
            summary += f"Your {y_col} is declining. Review recent changes and consider intervention strategies."
        else:
            summary += f"Your {y_col} is stable. Monitor for any sudden changes in upcoming periods."
        
        json_str = json.dumps(chart_json)
        return f"{summary}\n\n{json_str}"

    except Exception as e: return f"Forecasting Error: {str(e)}"


# --- 2. LEAD SCORING ENGINE (ENHANCED WITH EXPLAINABILITY) ---
def perform_lead_scoring(data_str: str):
    """Enhanced lead scoring with feature importance and recommendations"""
    try:
        data, error = safe_parse_data(data_str)
        if error: return error
        
        df = pd.DataFrame(data)
        required_cols = ['Status', 'Industry', 'LeadSource', 'AnnualRevenue', 'Rating']
        feature_cols = [c for c in required_cols if c in df.columns]
        
        if not feature_cols: return "Error: Missing columns (Industry, Revenue, etc)."

        # Fill missing values
        df[feature_cols] = df[feature_cols].fillna('Unknown')
        
        if 'AnnualRevenue' in df.columns:
            df['AnnualRevenue'] = df['AnnualRevenue'].replace('Unknown', 0).infer_objects(copy=False).astype(float)
            
            if df['AnnualRevenue'].mean() < 1000:
                return "**Insufficient Data** ⚠️\n\nCannot perform lead scoring: AnnualRevenue data is missing or incomplete."

        # Encode features
        le = LabelEncoder()
        feature_mapping = {}  # Store mappings for explainability
        
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                feature_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            else:
                df[f'{col}_encoded'] = df[col]

        encoded_cols = [f'{c}_encoded' for c in feature_cols]

        def get_target(status):
            s = str(status).lower()
            if 'qualified' in s or 'converted' in s or 'closed' in s: return 1
            if 'lost' in s or 'bad' in s: return 0
            return -1

        df_raw = pd.DataFrame(data)
        df['Target'] = df_raw['Status'].apply(get_target)

        train_df = df[df['Target'] != -1]
        predict_df = df[df['Target'] == -1]

        if len(train_df) < 1: 
             return "Not enough historical data (Qualified/Lost leads) to train."
        if len(predict_df) == 0: 
             return "No Open leads to score."

        X_train = train_df[encoded_cols]
        y_train = train_df['Target']
        
        unique_classes = y_train.unique()
        if len(unique_classes) < 2:
            return "**Insufficient Training Data** ⚠️\n\nNeed examples of both successful and unsuccessful leads."
        
        # Train model
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        
        # ✅ FEATURE IMPORTANCE (EINSTEIN-STYLE EXPLAINABILITY)
        importances = clf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        top_feature = feature_importance_df.iloc[0]['Feature']
        top_importance = feature_importance_df.iloc[0]['Importance'] * 100
        
        # Score leads
        probs = clf.predict_proba(predict_df[encoded_cols])[:, 1]
        
        results = df_raw.loc[predict_df.index].copy()
        results['Win_Probability'] = (probs * 100).round(1)
        top_leads = results.sort_values('Win_Probability', ascending=False).head(5)
        
        labels = top_leads['Name'].tolist() if 'Name' in top_leads.columns else [f"L{i}" for i in range(len(top_leads))]
        
        chart_json = {
            "chart_type": "bar",
            "title": "Top 5 High-Potential Leads (AI Scored)",
            "x_axis": "Lead Name",
            "y_axis": "Win Probability %",
            "data": [{"label": label, "value": value} for label, value in zip(labels, top_leads['Win_Probability'].tolist())]
        }
        
        # Build recommendations table
        table_md = "| Rank | Name | Company | Score | Action |\n|---|---|---|---|---|\n"
        for idx, (_, row) in enumerate(top_leads.iterrows(), 1):
            score = row['Win_Probability']
            if score > 75:
                action = "🔥 Contact ASAP"
            elif score > 50:
                action = "📞 Schedule call"
            else:
                action = "📧 Email nurture"
            
            table_md += f"| {idx} | {row.get('Name','N/A')} | {row.get('Company','N/A')} | **{score}%** | {action} |\n"

        # ✅ EINSTEIN-STYLE INSIGHTS
        summary = f"""🎯 **AI Lead Scoring Analysis**

**Model Performance:**
- Trained on {len(train_df)} historical leads
- **Top Predictive Factor:** {top_feature} ({top_importance:.1f}% influence)

**Top Opportunities Identified:**

{table_md}

**💡 Key Insights:**
"""
        
        high_prob_count = len(results[results['Win_Probability'] > 70])
        if high_prob_count > 0:
            summary += f"- You have **{high_prob_count} high-probability leads** (>70% score)\n"
            summary += f"- Focus sales efforts on top-scored leads for highest ROI\n"
        else:
            summary += "- No leads exceed 70% probability. Review lead quality sources.\n"
        
        summary += f"- **{top_feature}** is the strongest predictor of conversion\n"
        
        json_str = json.dumps(chart_json)
        return f"{summary}\n\n{json_str}"

    except Exception as e: return f"Scoring Error: {str(e)}"


# --- 3. CLUSTERING ENGINE (ENHANCED WITH PROFILING) ---
def perform_clustering(data_str: str):
    """Enhanced clustering with detailed segment profiling"""
    try:
        data, error = safe_parse_data(data_str)
        if error: return f"**Clustering Failed** ⚠️\n\n{error}"
        
        df = pd.DataFrame(data)
        
        if 'AnnualRevenue' not in df.columns:
            return "**Insufficient Data** ⚠️\n\nCannot perform clustering: AnnualRevenue field is missing."

        df['AnnualRevenue'] = df['AnnualRevenue'].replace([None, ''], 0).infer_objects(copy=False).astype(float)
        
        active_count = len(df[df['AnnualRevenue'] > 0])
        
        if active_count < 3:
            return f"**Insufficient Data** ⚠️\n\nNeed at least 3 records with valid revenue data. Currently: {active_count} records."

        # Map Rating
        rating_map = {'hot': 3, 'warm': 2, 'cold': 1}
        if 'Rating' in df.columns:
            df['Rating_Score'] = df['Rating'].astype(str).str.lower().map(rating_map).fillna(1)
        else:
            df['Rating_Score'] = 2

        X = df[['AnnualRevenue', 'Rating_Score']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # ✅ DETAILED CLUSTER PROFILING (EINSTEIN-STYLE)
        cluster_stats = df.groupby('Cluster').agg({
            'AnnualRevenue': ['mean', 'count', 'std'],
            'Rating_Score': 'mean'
        }).reset_index()
        
        cluster_stats.columns = ['Cluster', 'Avg_Revenue', 'Count', 'Revenue_Std', 'Avg_Rating']
        cluster_stats = cluster_stats.sort_values('Avg_Revenue', ascending=True)
        
        labels = ["🌱 Tier 3: Emerging", "📈 Tier 2: Growth", "💎 Tier 1: Enterprise"]
        cluster_stats['Label'] = labels[:len(cluster_stats)]
        
        chart_json = {
            "chart_type": "bar",
            "title": "Customer Segments by Revenue Tier",
            "x_axis": "Segment",
            "y_axis": "Average Revenue ($)",
            "data": [{"label": label, "value": value} for label, value in 
                     zip(cluster_stats['Label'].tolist(), cluster_stats['Avg_Revenue'].tolist())]
        }

        # Build detailed profile
        summary = f"""🧩 **Customer Segmentation Analysis**

I've analyzed your customer base and identified 3 distinct revenue tiers:

"""
        
        id_to_label = dict(zip(cluster_stats['Cluster'], cluster_stats['Label']))
        
        for _, row in cluster_stats.iterrows():
            segment_df = df[df['Cluster'] == row['Cluster']]
            avg_rating = row['Avg_Rating']
            
            if avg_rating >= 2.5:
                engagement = "High Engagement 🔥"
            elif avg_rating >= 1.5:
                engagement = "Medium Engagement ⚠️"
            else:
                engagement = "Low Engagement ❄️"
            
            summary += f"""**{row['Label']}**
- **Size:** {int(row['Count'])} customers ({row['Count']/len(df)*100:.1f}% of total)
- **Avg Revenue:** ${row['Avg_Revenue']:,.0f}
- **Engagement Level:** {engagement}

"""

        # ✅ ACTIONABLE RECOMMENDATIONS
        summary += """**💡 Recommended Actions:**

"""
        
        tier1_count = cluster_stats[cluster_stats['Label'].str.contains('Tier 1')]['Count'].values[0]
        tier3_count = cluster_stats[cluster_stats['Label'].str.contains('Tier 3')]['Count'].values[0]
        
        if tier1_count < len(df) * 0.2:
            summary += "- **Upsell Focus:** Only {:.0f}% are enterprise-tier. Identify growth opportunities in Tier 2.\n".format(tier1_count/len(df)*100)
        
        if tier3_count > len(df) * 0.4:
            summary += "- **Nurture Pipeline:** {:.0f}% are emerging customers. Develop nurture programs to accelerate growth.\n".format(tier3_count/len(df)*100)
        
        summary += "- **Resource Allocation:** Assign account managers proportionally to revenue tiers for maximum ROI.\n"

        json_str = json.dumps(chart_json)
        return f"{summary}\n\n{json_str}"

    except Exception as e:
        return f"Clustering Error: {str(e)}"


# --- 4. ✅ NEW: WHAT-IF ANALYSIS ENGINE ---
def perform_whatif_analysis(data_str: str, scenario: dict):
    """
    Perform what-if analysis by simulating changes to input variables.
    
    Args:
        data_str: Historical data
        scenario: Dict with changes, e.g., {"AnnualRevenue": "+20%", "Rating": "Hot"}
    
    Returns:
        Analysis showing predicted impact of changes
    """
    try:
        data, error = safe_parse_data(data_str)
        if error: return error
        
        df = pd.DataFrame(data)
        
        # Train baseline model
        required_cols = ['AnnualRevenue', 'Rating', 'Status']
        if not all(c in df.columns for c in required_cols):
            return "Error: Missing required columns for what-if analysis."
        
        # Encode features
        df['AnnualRevenue'] = pd.to_numeric(df['AnnualRevenue'], errors='coerce').fillna(0)
        rating_map = {'Hot': 3, 'Warm': 2, 'Cold': 1}
        df['Rating_Encoded'] = df['Rating'].map(rating_map).fillna(1)
        
        # Create target (conversion)
        df['Converted'] = df['Status'].str.contains('Qualified|Converted', case=False, na=False).astype(int)
        
        # Train model
        X = df[['AnnualRevenue', 'Rating_Encoded']]
        y = df['Converted']
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Calculate baseline conversion rate
        baseline_conversion = y.mean() * 100
        
        # Apply scenario changes
        df_scenario = df.copy()
        
        for feature, change in scenario.items():
            if feature == 'AnnualRevenue' and isinstance(change, str) and '%' in change:
                # Percentage change
                pct = float(change.replace('%', '').replace('+', ''))
                df_scenario['AnnualRevenue'] = df_scenario['AnnualRevenue'] * (1 + pct/100)
            elif feature == 'Rating':
                df_scenario['Rating_Encoded'] = rating_map.get(change, 2)
        
        # Predict with new scenario
        X_scenario = df_scenario[['AnnualRevenue', 'Rating_Encoded']]
        scenario_predictions = model.predict(X_scenario)
        scenario_conversion = scenario_predictions.mean() * 100
        
        # Calculate impact
        impact = scenario_conversion - baseline_conversion
        impact_pct = (impact / baseline_conversion * 100) if baseline_conversion > 0 else 0
        
        summary = f"""🔮 **What-If Analysis Results**

**Scenario Applied:**
"""
        for feature, change in scenario.items():
            summary += f"- {feature}: {change}\n"
        
        summary += f"""
**Predicted Impact:**
- **Baseline Conversion Rate:** {baseline_conversion:.1f}%
- **Scenario Conversion Rate:** {scenario_conversion:.1f}%
- **Expected Change:** {'+' if impact > 0 else ''}{impact:.1f} percentage points ({'+' if impact_pct > 0 else ''}{impact_pct:.1f}%)

**💡 Interpretation:**
"""
        
        if impact > 5:
            summary += f"This change would likely **increase conversions significantly**. Consider implementing this strategy."
        elif impact > 0:
            summary += f"This change would have a **modest positive impact** on conversions."
        elif impact < -5:
            summary += f"⚠️ This change could **decrease conversions**. Reconsider this approach."
        else:
            summary += f"This change would have **minimal impact** on conversion rates."
        
        return summary
        
    except Exception as e:
        return f"What-If Analysis Error: {str(e)}"


# --- 5. ✅ NEW: ANOMALY DETECTION ENGINE ---
def detect_anomalies(data_str: str, threshold: float = 2.0):
    """
    Detect anomalies in time-series data using statistical methods.
    
    Args:
        data_str: Time-series data
        threshold: Number of standard deviations to flag as anomaly
    
    Returns:
        List of anomalous periods with explanations
    """
    try:
        data, error = safe_parse_data(data_str)
        if error: return error
        
        df = pd.DataFrame(data)
        cols = df.columns.tolist()
        
        if len(cols) < 2:
            return "Error: Need at least date and value columns."
        
        # Parse dates
        if len(cols) >= 3:
            df['Date'] = pd.to_datetime(dict(year=df[cols[0]], month=df[cols[1]], day=1))
            y_col = cols[2]
        else:
            df['Date'] = pd.to_datetime(df[cols[0]])
            y_col = cols[1]
        
        df = df.sort_values('Date')
        
        # Calculate rolling statistics
        df['Mean'] = df[y_col].rolling(window=3, min_periods=1).mean()
        df['Std'] = df[y_col].rolling(window=3, min_periods=1).std()
        
        # Detect anomalies (beyond threshold standard deviations)
        df['Z_Score'] = (df[y_col] - df['Mean']) / (df['Std'] + 1e-9)  # Avoid division by zero
        df['Is_Anomaly'] = df['Z_Score'].abs() > threshold
        
        anomalies = df[df['Is_Anomaly']]
        
        if len(anomalies) == 0:
            return f"✅ **No Anomalies Detected**\n\nAll {len(df)} data points fall within expected range (±{threshold} standard deviations)."
        
        summary = f"""⚠️ **Anomaly Detection Results**

**{len(anomalies)} anomalous period(s) detected** out of {len(df)} total periods:

"""
        
        for _, row in anomalies.iterrows():
            date_str = row['Date'].strftime('%Y-%m')
            value = row[y_col]
            z_score = row['Z_Score']
            
            if z_score > 0:
                direction = "📈 Unusually High"
            else:
                direction = "📉 Unusually Low"
            
            summary += f"- **{date_str}**: {value:,.1f} ({direction}, {abs(z_score):.1f}σ from mean)\n"
        
        summary += f"""
**💡 Recommendations:**
- Investigate these periods for unusual events or data quality issues
- Consider seasonal factors or business changes during anomalous periods
- Update forecasts if anomalies represent new normal trends
"""
        
        return summary
        
    except Exception as e:
        return f"Anomaly Detection Error: {str(e)}"
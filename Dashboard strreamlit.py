import os
# FIX 1: Force TensorFlow to use the CPU. This prevents CUDA errors and instability.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pydeck as pdk

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Consumption Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)
st.title("⚡ Energy Consumption Forecast Dashboard")

# --- Path Configuration and Model Mapping ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "powerconsumption.csv")

MODEL_MAP = {
    "Zone 1 – Marina Smir (Commercial/Touristic)": "PowerConsumption_Zone1",
    "Zone 2 – Boussafou (Residential)": "PowerConsumption_Zone2",
    "Zone 3 – M’hannech (Industrial/Utility)": "PowerConsumption_Zone3"
}
ZONE_COORDINATES = {
    "Zone 1 – Marina Smir (Commercial/Touristic)": (35.731, -5.335),
    "Zone 2 – Boussafou (Residential)": (35.588, -5.350),
    "Zone 3 – M’hannech (Industrial/Utility)": (35.584, -5.363)
}

# --- Initialize Session State for Tariff Comparison ---
if 'baseline_peak_rate' not in st.session_state:
    st.session_state.baseline_peak_rate = 0.30
if 'baseline_offpeak_rate' not in st.session_state:
    st.session_state.baseline_offpeak_rate = 0.15

# --- Helper Functions ---
@st.cache_data
def load_data(path):
    """Loads and preprocesses the data."""
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        return None
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    return df

@st.cache_data
def feature_engineer(df):
    """Creates engineered features for the model."""
    df_eng = df.copy()
    df_eng['hour'] = df_eng.index.hour
    df_eng['day_of_week'] = df_eng.index.dayofweek
    df_eng['month'] = df_eng.index.month
    df_eng['season'] = (df_eng['month'] % 12 // 3) + 1
    df_eng['hourly_cos'] = np.cos(2 * np.pi * df_eng['hour'] / 24)
    df_eng['is_weekend'] = (df_eng.index.dayofweek >= 5).astype(int)
    df_eng['daytype'] = np.where(df_eng['day_of_week'] < 5, 1, 0)
    df_eng['temperature'] = np.random.normal(22, 5, len(df_eng))
    df_eng['humidity'] = np.random.normal(60, 15, len(df_eng))
    df_eng['wind'] = np.random.normal(10, 5, len(df_eng))
    moroccan_holidays_2017 = [
        "2017-01-01", "2017-01-11", "2017-05-01", "2017-06-26", "2017-07-30",
        "2017-08-14", "2017-08-20", "2017-08-21", "2017-09-02", "2017-09-22",
        "2017-11-06", "2017-11-18", "2017-12-01"
    ]
    df_eng['is_holiday'] = df_eng.index.normalize().isin(pd.to_datetime(moroccan_holidays_2017)).astype(int)
    df_eng['school_holiday'] = 0
    for zone in MODEL_MAP.values():
        for lag in [1, 6, 12, 144]:
            df_eng[f"{zone}_lag_{lag}"] = df_eng[zone].shift(lag)
        df_eng[f"{zone}_roll_avg_{144}"] = df_eng[zone].rolling(window=144, min_periods=1).mean()
    return df_eng.dropna()

def prepare_sequences(df, zone):
    """Prepares data sequences for the LSTM model."""
    cols_to_drop = list(MODEL_MAP.values())
    X = df.drop(columns=cols_to_drop)
    y = df[[zone]]
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)
    LOOK_BACK = 144
    X_seq, y_seq, y_idx = [], [], y.index[LOOK_BACK:]
    for i in range(len(X_scaled) - LOOK_BACK):
        X_seq.append(X_scaled[i:i+LOOK_BACK])
        y_seq.append(y_scaled[i + LOOK_BACK])
    return np.array(X_seq), np.array(y_seq), scaler_y, y_idx, X.columns

# --- Main Application Logic ---
data_raw = load_data(DATA_FILE)
if data_raw is not None:
    data_engineered = feature_engineer(data_raw)

    # --- Sidebar ---
    st.sidebar.header("📅 Filters")
    min_date = data_engineered.index.min().date()
    max_date = data_engineered.index.max().date()

    # FIX 2: Set a smaller default date range to prevent freezing on load.
    default_start_date = max_date - pd.Timedelta(days=30)
    if default_start_date < min_date:
        default_start_date = min_date

    start_date = st.sidebar.date_input("Start Date", value=default_start_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    st.sidebar.info("💡 Tip: Selecting a smaller date range will improve app performance.")

    if start_date > end_date:
        st.sidebar.warning("Start date must be before end date.")
        st.stop()
    
    filtered_data = data_engineered.loc[str(start_date):str(end_date)]

    st.sidebar.header("📊 Aggregation")
    # FIX 3: Default to "Daily" aggregation for better performance.
    agg_level = st.sidebar.radio("View by:", ["Hourly", "Daily", "Weekly"], index=1, horizontal=True)

    st.sidebar.header("⚡ Tariff Controls")
    peak_rate = st.sidebar.slider("Peak Tariff (£/kWh)", 0.0, 1.00, 0.30, step=0.01)
    offpeak_rate = st.sidebar.slider("Off-Peak Tariff (£/kWh)", 0.0, 0.50, 0.15, step=0.01)

    st.sidebar.header("🚨 Anomaly Sensitivity")
    anomaly_std = st.sidebar.slider("Sensitivity (Std. Dev.)", 1.0, 6.0, 4.0, step=0.1)

    st.sidebar.header("⚖️ What-If Simulation")
    shift_percentage = st.sidebar.slider(
        "Shift Peak Usage to Off-Peak (%)", 0, 100, 10, 5,
        help="Simulate the financial impact of shifting a percentage of peak-hour consumption to off-peak hours."
    )

    # --- Main Zone Selection ---
    st.markdown("### Select Primary Zone for Analysis")
    main_zone_label = st.selectbox("Primary Zone:", options=list(MODEL_MAP.keys()), index=0)
    main_zone_name = MODEL_MAP[main_zone_label]

    # --- Tabs ---
    tabs = st.tabs(["📈 Forecast", "💷 Tariff", "📊 Summary", "🗺️ Map", "🔁 Compare Zones", "💡 Model Performance", "🧠 Feature Importance"])
    forecast_tab, tariff_tab, summary_tab, map_tab, compare_tab, explanation_tab, importance_tab = tabs

    # --- OPTIMIZATION: Pre-calculate predictions for all zones ---
    @st.cache_data(show_spinner="Generating all zone forecasts...")
    def get_all_predictions(_data):
        all_results = {}
        for label, zone_name in MODEL_MAP.items():
            model_path = os.path.join(BASE_DIR, f"lstm_model_{zone_name}_32f.keras")
            if not os.path.exists(model_path):
                all_results[label] = (None, f"Model file not found: {model_path}")
                continue
            
            model = load_model(model_path)
            X_seq, y_seq, scaler_y, idx, feature_names = prepare_sequences(_data, zone_name)
            
            if model.input_shape[-1] != X_seq.shape[-1]:
                 all_results[label] = (None, f"Feature mismatch! Model expects {model.input_shape[-1]}, got {X_seq.shape[-1]}.")
                 continue

            pred_scaled = model.predict(X_seq)
            pred_values = scaler_y.inverse_transform(pred_scaled)
            actual_values = scaler_y.inverse_transform(y_seq)
            results_df = pd.DataFrame({"Actual": actual_values.flatten(), "Predicted": pred_values.flatten()}, index=idx)
            all_results[label] = (results_df, None)
        return all_results

    all_zone_results = get_all_predictions(filtered_data.copy())
    
    # --- Process main selected zone from pre-calculated results ---
    main_results, error = all_zone_results[main_zone_label]

    if error:
        st.error(error)
        st.stop()

    main_results = main_results.join(filtered_data[['season', 'hour', 'day_of_week']])
    main_results["error"] = main_results["Actual"] - main_results["Predicted"]
    main_results["Anomaly"] = abs(main_results["error"]) > (main_results["error"].std() * anomaly_std)
    
    # --- Resample data based on sidebar selection ---
    def resample_df(df, level):
        if level == "Daily": return df.resample("D").mean()
        if level == "Weekly": return df.resample("W").mean()
        return df

    display_df = resample_df(main_results.copy(), agg_level)

    # --- Display Tabs ---
    with forecast_tab:
        st.subheader(f"Forecast for {main_zone_label}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=display_df.index, y=display_df["Actual"], name="Actual Consumption", line=dict(color='#3366CC')))
        fig.add_trace(go.Scatter(x=display_df.index, y=display_df["Predicted"], name="Predicted Consumption", line=dict(color='#999999', dash='dot')))
        anomalies_to_plot = main_results[main_results["Anomaly"]]
        if not anomalies_to_plot.empty:
            fig.add_trace(go.Scatter(x=anomalies_to_plot.index, y=anomalies_to_plot["Actual"], name="Anomalies", mode="markers", marker=dict(color="#DC3912", size=8, symbol='x')))
        fig.update_layout(height=500, title="Consumption Forecast with Anomaly Detection", yaxis_title="Consumption (kW)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Deeper Error Analysis")
        error_fig = go.Figure(go.Histogram(x=main_results['error'], nbinsx=50, name="Prediction Error"))
        error_fig.update_layout(title="Distribution of Prediction Errors (Actual - Predicted)", xaxis_title="Error (kW)", yaxis_title="Frequency")
        st.plotly_chart(error_fig, use_container_width=True)


    with tariff_tab:
        st.subheader(f"Tariff Cost Simulation ({main_zone_label})")
        main_results["is_peak"] = ((main_results.index.hour >= 8) & (main_results.index.hour < 18) & (main_results.index.dayofweek < 5))
        
        st.markdown("##### Baseline & Current Cost")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###### Current Simulation")
            current_actual_cost = (main_results["Actual"] * (1/6) * np.where(main_results["is_peak"], peak_rate, offpeak_rate)).sum()
            current_predicted_cost = (main_results["Predicted"] * (1/6) * np.where(main_results["is_peak"], peak_rate, offpeak_rate)).sum()
            st.metric("Actual Cost", f"£{current_actual_cost:,.2f}")
            st.metric("Predicted Cost", f"£{current_predicted_cost:,.2f}")
        with col2:
            st.markdown("###### Baseline Comparison")
            baseline_actual_cost = (main_results["Actual"] * (1/6) * np.where(main_results["is_peak"], st.session_state.baseline_peak_rate, st.session_state.baseline_offpeak_rate)).sum()
            cost_change = current_actual_cost - baseline_actual_cost
            st.metric(f"Baseline Cost (at £{st.session_state.baseline_peak_rate:.2f}/£{st.session_state.baseline_offpeak_rate:.2f})", f"£{baseline_actual_cost:,.2f}")
            st.metric("Change from Baseline", f"£{cost_change:,.2f}", delta=f"{cost_change:,.2f}", delta_color="inverse")
        if st.button("Set Current as Baseline"):
            st.session_state.baseline_peak_rate, st.session_state.baseline_offpeak_rate = peak_rate, offpeak_rate
            st.success("Baseline tariffs updated!")
            st.rerun()
        
        st.markdown("---")
        st.subheader("What-If: Peak-to-Off-Peak Shift Simulation")
        if shift_percentage > 0:
            peak_consumption_df = main_results[main_results["is_peak"]]
            total_peak_kwh = peak_consumption_df['Actual'].sum()
            kwh_to_shift = total_peak_kwh * (shift_percentage / 100.0)
            cost_saving_from_shift = kwh_to_shift * (peak_rate - offpeak_rate)
            new_simulated_cost = current_actual_cost - cost_saving_from_shift
            st.markdown(f"By shifting **{shift_percentage}%** of peak usage ({kwh_to_shift:,.2f} kWh) to off-peak hours, you could achieve the following savings:")
            col1, col2 = st.columns(2)
            col1.metric("New Estimated Cost for Period", f"£{new_simulated_cost:,.2f}", delta=f"-£{cost_saving_from_shift:,.2f}", delta_color="normal")
            days_in_period = (filtered_data.index.max() - filtered_data.index.min()).days + 1
            if days_in_period > 0:
                annual_savings_estimate = (cost_saving_from_shift / days_in_period) * 365
                col2.metric("Estimated Annual Savings", f"£{annual_savings_estimate:,.2f}")
        else:
            st.info("Move the 'Shift Peak Usage' slider in the sidebar to simulate cost savings.")


    with summary_tab:
        st.subheader(f"Consumption Summary ({main_zone_label})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Consumption", f"{main_results['Actual'].sum():,.2f} kWh")
        col2.metric("Average Daily Consumption", f"{main_results['Actual'].resample('D').sum().mean():,.2f} kWh")
        col3.metric("Peak Value", f"{main_results['Actual'].max():.2f} kW")
        
        st.markdown("---")
        st.subheader("Consumption Patterns")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Average by Hour of Day")
            hourly_usage = main_results.groupby('hour')['Actual'].mean()
            fig_hourly = go.Figure(go.Bar(x=hourly_usage.index, y=hourly_usage.values, marker_color='#3366CC'))
            fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Consumption (kW)")
            st.plotly_chart(fig_hourly, use_container_width=True)
        with col2:
            st.markdown("##### Average by Day of Week")
            day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            daily_usage = main_results.groupby('day_of_week')['Actual'].mean().rename(index=day_map).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig_daily = go.Figure(go.Bar(x=daily_usage.index, y=daily_usage.values, marker_color='#FF9900'))
            fig_daily.update_layout(xaxis_title="Day of Week", yaxis_title="Average Consumption (kW)")
            st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("---")
        st.subheader("Cross-Zone Comparison")
        summary_data = []
        for label, name in MODEL_MAP.items():
            zone_data = filtered_data[[name]].copy()
            summary_data.append({
                "Zone": label,
                "Total Consumption (kWh)": zone_data[name].sum(),
                "Average Daily Consumption (kWh)": zone_data[name].resample('D').sum().mean(),
                "Peak Value (kW)": zone_data[name].max()
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.style.format({
            "Total Consumption (kWh)": "{:,.2f}",
            "Average Daily Consumption (kWh)": "{:,.2f}",
            "Peak Value (kW)": "{:,.2f}"
        }))

        st.markdown("---")
        st.subheader("Export Data")
        st.download_button(
            label="Download Full Results as CSV",
            data=main_results.to_csv().encode('utf-8'),
            file_name=f'forecast_results_{main_zone_name}.csv',
            mime='text/csv',
        )


    with map_tab:
        st.subheader("Zone Map")
        map_df = pd.DataFrame.from_dict(ZONE_COORDINATES, orient='index', columns=['lat', 'lon'])
        map_df['zone_name'] = map_df.index
        view_state = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=10, pitch=50)
        layer = pdk.Layer("ScatterplotLayer", data=map_df, get_position=["lon", "lat"], get_color=[200, 30, 0, 160], get_radius=200, pickable=True)
        tooltip = {"html": "<b>Zone:</b> {zone_name}"}
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9", tooltip=tooltip)
        st.pydeck_chart(r)

    with compare_tab:
        st.subheader(f"Compare {main_zone_label} with:")
        zones_to_compare = {label: name for label, name in MODEL_MAP.items() if name != main_zone_label}
        compare_labels = st.multiselect("Select zones to compare", list(zones_to_compare.keys()))
        
        if compare_labels:
            fig2 = go.Figure()
            # Plot main zone
            main_compare_df = resample_df(main_results.copy(), agg_level)
            fig2.add_trace(go.Scatter(x=main_compare_df.index, y=main_compare_df["Actual"], name=main_zone_label, line=dict(color='#3366CC')))
            
            color_palette = ['#109618', '#990099'] # Green, Purple
            for i, label in enumerate(compare_labels):
                # OPTIMIZATION: Use pre-calculated results instead of re-calculating
                compare_results, compare_error = all_zone_results[label]
                if compare_error:
                    st.warning(f"Could not load data for {label}: {compare_error}")
                else:
                    compare_display_df = resample_df(compare_results.copy(), agg_level)
                    fig2.add_trace(go.Scatter(x=compare_display_df.index, y=compare_display_df["Actual"], name=label, line=dict(color=color_palette[i % len(color_palette)])))
            
            fig2.update_layout(title="Zone Comparison", height=500, yaxis_title="Consumption (kW)")
            st.plotly_chart(fig2, use_container_width=True)

    with explanation_tab:
        st.subheader("How Good is This Forecast?")
        st.markdown("""
        This dashboard uses a sophisticated computer model (a type of AI called an LSTM) to predict energy usage. But how can you tell if it's doing a good job? Here's a simple guide.
        #### 1. The Forecast Chart (📈 Forecast Tab)
        The main chart shows two important lines:
        - **Actual Consumption:** This is the real amount of energy that was used.
        - **Predicted Consumption:** This is what our computer model *thought* would happen based on historical patterns.
        **What to look for:** A good forecast means the 'Predicted' line stays very close to the 'Actual' line.
        #### 2. The Red 'X' Marks (Anomalies)
        These are not mistakes! They are moments the model found interesting because the *actual* energy use was very different from what was *predicted*. This helps city planners spot potential issues like power outages or equipment failures.
        You can use the **"Anomaly Sensitivity"** slider in the sidebar to control how many of these points are flagged.
        #### 3. The Cost Simulation (💷 Tariff Tab)
        This tab shows the estimated electricity cost based on the model's predictions versus the cost of what actually happened. A small difference means the model is reliable enough for financial planning.
        ---
        **Overall Verdict:** This model is a powerful tool for understanding and forecasting energy consumption. It's accurate enough for strategic planning, budget forecasting, and identifying unusual events in the city's power grid.
        """)
    
    with importance_tab:
        st.subheader("Which Factors Influence the Forecast the Most?")
        st.markdown("""
        This chart shows which pieces of information (features) are most important to the model when it makes a prediction. A higher score means the model relies on that feature more. This helps us understand *why* the model makes certain decisions.
        """)

        @st.cache_data(show_spinner="Calculating feature importances...")
        def get_feature_importance(_data, _zone_name):
            cols_to_drop = list(MODEL_MAP.values())
            X = _data.drop(columns=cols_to_drop)
            y = _data[_zone_name]
            
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            return importance_df

        importance_df = get_feature_importance(filtered_data.copy(), main_zone_name)

        fig_importance = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h'
        ))
        fig_importance.update_layout(
            title=f"Top 15 Most Important Features for {main_zone_label}",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)

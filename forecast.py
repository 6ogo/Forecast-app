import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import chardet
import io

# App title
st.title("Interactive Forecasting Dashboard")

# Sidebar for user inputs
st.sidebar.header("Data & Forecast Settings")

# File uploader for both CSV and XLSX
uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Determine file type from extension
    file_type = uploaded_file.name.split('.')[-1].lower()

    # Sidebar options for encoding and separator
    st.sidebar.subheader("File Settings")
    encoding = st.sidebar.selectbox("Encoding", ["Auto", "utf-8", "latin1", "iso-8859-1", "cp1252"], index=0)
    if file_type == "csv":
        separator = st.sidebar.selectbox("Separator", ["Auto", ",", ";", "\t", "|"], index=0)
    else:
        separator = None  # Separator not needed for XLSX

    # Function to detect encoding automatically
    def detect_encoding(file):
        raw_data = file.read(10000)  # Read first 10k bytes
        file.seek(0)  # Reset file pointer
        result = chardet.detect(raw_data)
        return result['encoding']

    # Function to detect separator for CSV automatically
    def detect_separator(file):
        sample = file.read(1024).decode('utf-8', errors='ignore')  # Read first 1024 bytes
        file.seek(0)  # Reset file pointer
        if sample.count(',') > sample.count(';'):
            return ','
        elif sample.count(';') > sample.count(','):
            return ';'
        elif sample.count('\t') > sample.count(','):
            return '\t'
        else:
            return ','  # Default to comma

    # Load and cache the data
    @st.cache_data
    def load_data(file, file_type, encoding, separator):
        # Handle automatic encoding detection
        if encoding == "Auto":
            encoding = detect_encoding(file)
        
        # Load data based on file type
        if file_type == "csv":
            if separator == "Auto":
                separator = detect_separator(file)
            data = pd.read_csv(file, encoding=encoding, sep=separator)
        elif file_type == "xlsx":
            data = pd.read_excel(file, engine='openpyxl')
        return data

    # Load the data with specified or auto-detected settings
    df = load_data(uploaded_file, file_type, encoding, separator)
    st.write("Loaded Data:", df.head())

    # Column selection
    st.sidebar.subheader("Column Selection")
    columns = df.columns.tolist()
    mode = st.sidebar.radio("Column selection mode", ("Manual", "Auto"))

    if mode == "Manual":
        date_col = st.sidebar.selectbox("Select date column", columns)
        value_col = st.sidebar.selectbox("Select value column", columns)
    else:  # Auto mode
        # Assume first column with date-like data is 'ds', next numeric is 'y'
        date_col = next(col for col in columns if pd.to_datetime(df[col], errors='coerce').notna().sum() > len(df) * 0.9)
        value_col = next(col for col in columns if pd.api.types.is_numeric_dtype(df[col]) and col != date_col)
        st.sidebar.write(f"Auto-detected: Date = {date_col}, Value = {value_col}")

    # Prepare data for Prophet
    df_prophet = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')

    if df_prophet['ds'].isna().any():
        st.error("Some dates could not be parsed. Please ensure the date column is in a valid format.")
    else:
        # Infer frequency
        df_prophet = df_prophet.set_index('ds')
        freq = pd.infer_freq(df_prophet.index)
        if freq is None:
            st.warning("Frequency not inferred. Defaulting to daily.")
            freq = 'D'
        df_prophet = df_prophet.reset_index()

        # Train the model
        @st.cache_resource
        def train_model(df):
            model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
            model.fit(df)
            return model

        model = train_model(df_prophet)

        # Forecast settings
        st.sidebar.subheader("Forecast Options")
        max_periods = 365
        periods = st.sidebar.slider("Forecast periods", 1, max_periods, 30, help="Adjust how far into the future to predict")
        show_fitted = st.sidebar.checkbox("Show fitted values", help="Display model fit on historical data")

        # Generate future dataframe and predict
        future = model.make_future_dataframe(periods=max_periods, freq=freq)
        forecast = model.predict(future)

        # Separate historical and future data
        last_historical_date = df_prophet['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_historical_date].head(periods)

        # Create interactive Plotly chart
        fig = go.Figure()

        # Historical data (solid line)
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'], 
            y=df_prophet['y'], 
            mode='lines', 
            name='Historical', 
            line=dict(color='#1f77b4', width=2)
        ))

        # Forecast (dotted line)
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], 
            y=future_forecast['yhat'], 
            mode='lines', 
            name='Forecast', 
            line=dict(color='#ff7f0e', width=2, dash='dot')
        ))

        # Confidence intervals (shaded area)
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Vertical line at forecast start
        fig.add_vline(
            x=last_historical_date, 
            line_width=2, 
            line_dash="dash", 
            line_color="#2ca02c"
        )

        # Optional: Fitted values
        if show_fitted:
            fitted = forecast[forecast['ds'] <= last_historical_date]
            fig.add_trace(go.Scatter(
                x=fitted['ds'], 
                y=fitted['yhat'], 
                mode='lines', 
                name='Fitted', 
                line=dict(color='#9467bd', width=1.5)
            ))

        # Customize layout
        fig.update_layout(
            title="Data Forecast",
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Legend",
            template="plotly_white",
            font=dict(size=14),
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True
        )

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table and download
        st.subheader("Forecast Results")
        display_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'}
        )
        st.dataframe(display_forecast)

        csv = display_forecast.to_csv(index=False)
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv"
        )
else:
    st.write("Please upload a file to begin forecasting.")
import streamlit as st
import pandas as pd
import numpy as np
import pickle

try:
    import interpret
except ImportError:
    st.warning("'interpret' library is not installed. Model explanations will be disabled.")

st.set_page_config(layout="wide")

st.title('DAYTON_MW Forecasting with Explainable Boosting Machine (EBM)')

# --- 1. Load the Trained Model ---
model_filename = 'ebm_model.pkl'

try:
    with open(model_filename, 'rb') as file:
        ebm_model = pickle.load(file)
    st.success(f"EBM model loaded successfully from {model_filename}")
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the correct directory.")
    st.stop() # Stop the app if the model can't be loaded

# --- 2. Create Input Fields for Features ---
st.header('Input Features for Prediction')

with st.expander("Temporal Features", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hour = st.slider('Hour', 0, 23, 12)
    with col2:
        dayofweek = st.slider('Day of Week', 0, 6, 4) # Monday=0, Sunday=6
    with col3:
        month = st.slider('Month', 1, 12, 7)
    with col4:
        year = st.slider('Year', 2004, 2018, 2017) # Based on data range

with st.expander("Lag Features (Previous DAYTON_MW values)", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        lag_1h = st.number_input('Lag 1 Hour', value=1800.0, format="%.1f", help="MW value from 1 hour ago")
    with col2:
        lag_24h = st.number_input('Lag 24 Hours', value=1750.0, format="%.1f", help="MW value from 24 hours ago")
    with col3:
        lag_48h = st.number_input('Lag 48 Hours', value=1700.0, format="%.1f", help="MW value from 48 hours ago")
    with col4:
        lag_168h = st.number_input('Lag 168 Hours', value=1600.0, format="%.1f", help="MW value from 1 week ago")
    with col5:
        lag_336h = st.number_input('Lag 336 Hours', value=1500.0, format="%.1f", help="MW value from 2 weeks ago")

with st.expander("Rolling Mean Features (Average DAYTON_MW values)", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        rolling_mean_24h = st.number_input('Rolling Mean 24 Hours', value=1800.0, format="%.1f", help="Average MW over the last 24 hours")
    with col2:
        rolling_mean_48h = st.number_input('Rolling Mean 48 Hours', value=1750.0, format="%.1f", help="Average MW over the last 48 hours")
    with col3:
        rolling_mean_168h = st.number_input('Rolling Mean 168 Hours', value=1700.0, format="%.1f", help="Average MW over the last 168 hours")

# --- 3. Make Predictions with User Input ---
if st.button('Predict DAYTON_MW'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[hour, dayofweek, month, year,
                                  lag_1h, lag_24h, lag_48h, lag_168h, lag_336h,
                                  rolling_mean_24h, rolling_mean_48h, rolling_mean_168h]],
                                columns=['hour', 'dayofweek', 'month', 'year',
                                         'lag_1h', 'lag_24h', 'lag_48h', 'lag_168h', 'lag_336h',
                                         'rolling_mean_24h', 'rolling_mean_48h', 'rolling_mean_168h'])

    # Make prediction
    prediction = ebm_model.predict(input_data)[0]

    st.subheader('Prediction Result:')
    st.metric(label="Predicted DAYTON_MW", value=f"{prediction:.2f} MW")

    st.info("Adjust the input features above to see how the prediction changes.")

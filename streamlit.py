import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("D:\CO2FORE\co2_mm_mlo.csv")
    monthly_data = data[['average']]
    monthly_data.index = pd.to_datetime(data[['year', 'month']].assign(day=1))
    return monthly_data

monthly_data = load_data()

# Sidebar options
st.sidebar.header("Model Parameters")

# Model selection
model_options = st.sidebar.multiselect("Select Models", ["MA", "ARMA", "MLP"], default=["MA", "ARMA", "MLP"])

# Parameters for MA
if "MA" in model_options:
    K = st.sidebar.slider("K (number of past readings) for MA", min_value=1, max_value=12, value=3)

# Parameters for ARMA
if "ARMA" in model_options:
    p = st.sidebar.slider("AR order (p) for ARMA", min_value=1, max_value=5, value=1)
    q = st.sidebar.slider("MA order (q) for ARMA", min_value=1, max_value=5, value=1)

# Parameters for MLP
if "MLP" in model_options:
    mlp_param_search = st.sidebar.checkbox("Find Best Parameters for MLP")
    if mlp_param_search:
        K = st.sidebar.slider("K (number of past readings) for MLP", min_value=1, max_value=12, value=3)
        T = st.sidebar.slider("T (forecast horizon) for MLP", min_value=1, max_value=3, value=1)

# Date range selection
date_range = st.sidebar.date_input("Select Date Range", value=[monthly_data.index.min(), monthly_data.index.max()])
monthly_data = monthly_data[(monthly_data.index >= pd.to_datetime(date_range[0])) & 
                            (monthly_data.index <= pd.to_datetime(date_range[1]))]

# Forecast horizon
forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", min_value=1, max_value=24, value=12)

# Split the dataset into training and testing sets
train_size = int(len(monthly_data) * 0.8)
train_data, test_data = monthly_data[:train_size], monthly_data[train_size+1:]

# Define functions for MA, ARMA, and MLP models
def moving_average(train, test, k):
    predictions = []
    history = list(train[-k:])
    for t in range(len(test)):
        yhat = np.mean(history)
        predictions.append(yhat)
        history.append(test[t])
    return predictions

def arma(train, test, p, q):
    predictions = []
    history = list(train)
    for t in range(len(test)):
        model = ARIMA(history, order=(p, 0, q))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    return predictions

def mlp(train, test, k):
    X_train, y_train = [], []
    for i in range(len(train) - k):
        X_train.append(train[i:i+k])
        y_train.append(train[i+k])
    X_test = [test[i:i+k] for i in range(len(test) - k)]
    
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
    mlp_regressor.fit(X_train, y_train)
    predictions = mlp_regressor.predict(X_test)
    return predictions

def find_best_mlp_params(train_values, k_folds=5):
    param_grid = {
        'K': [3, 6, 9, 12],
        'T': [1, 2, 3]
    }

    best_rmse = float('inf')
    best_params = {}

    kf = KFold(n_splits=k_folds)

    for k_param in param_grid['K']:
        for t_param in param_grid['T']:
            rmse_sum = 0
            for train_index, val_index in kf.split(train_values):
                X_train_fold, X_val_fold = train_values[train_index], train_values[val_index]
                mlp_forecasts = mlp(X_train_fold, X_val_fold, k_param)
                rmse_fold = np.sqrt(mean_squared_error(X_val_fold[:len(mlp_forecasts)], mlp_forecasts))
                rmse_sum += rmse_fold
            avg_rmse = rmse_sum / k_folds
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params['K'] = k_param
                best_params['T'] = t_param

    return best_params

train_values = train_data['average'].values
test_values = test_data['average'].values

st.title("Mauna Loa CO2 Forecasting")

# Find the best parameters for MLP using k-fold cross-validation
if "MLP" in model_options and mlp_param_search:
    best_params = find_best_mlp_params(train_values)
    K = best_params['K']
    T = best_params['T']
    st.write("Best parameters for MLP:", best_params)
else:
    K = st.sidebar.slider("K (number of past readings) for MLP", min_value=1, max_value=12, value=3)
    T = st.sidebar.slider("T (forecast horizon) for MLP", min_value=1, max_value=3, value=1)

# Generate forecasts
forecasts = {}
if "MA" in model_options:
    forecasts['MA'] = moving_average(train_values, test_values, K)
if "ARMA" in model_options:
    forecasts['ARMA'] = arma(train_values, test_values, p, q)
if "MLP" in model_options:
    forecasts['MLP'] = mlp(train_values, test_values, K)

# Evaluate forecasts using RMSE
results = {}
if "MA" in model_options:
    ma_rmse = np.sqrt(mean_squared_error(test_values[:len(forecasts['MA'])], forecasts['MA']))
    results['MA'] = ma_rmse
if "ARMA" in model_options:
    arma_rmse = np.sqrt(mean_squared_error(test_values[:len(forecasts['ARMA'])], forecasts['ARMA']))
    results['ARMA'] = arma_rmse
if "MLP" in model_options:
    mlp_rmse = np.sqrt(mean_squared_error(test_values[:len(forecasts['MLP'])], forecasts['MLP']))
    results['MLP'] = mlp_rmse

st.write("RMSE for Models:")
st.write(results)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[:len(test_values)], test_values, label='Actual')
for model_name, model_forecasts in forecasts.items():
    plt.plot(test_data.index[:len(model_forecasts)], model_forecasts, label=model_name)
plt.legend()
st.pyplot(plt)

# Download results
if st.sidebar.button("Download Results as CSV"):
    results_df = pd.DataFrame(results, index=[0])
    st.download_button("Download CSV", data=results_df.to_csv(index=False), file_name="forecast_results.csv")

# Display data insights
if st.sidebar.checkbox("Show Data Insights"):
    st.write("Data Summary")
    st.write(monthly_data.describe())

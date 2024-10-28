import pandas as pd
import numpy as np
import os.path as osp
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Define the data root and hand features
data_root = '/mnt/c/Users/Gomzi/tropicalcyclone_MLP/hurricane_data/'
hand_features = ['vs0', 'PSLV_v2', 'PSLV_v3', 'PSLV_v4', 'PSLV_v5', 'PSLV_v6', 'PSLV_v7',
                 'PSLV_v8', 'PSLV_v9', 'PSLV_v10', 'PSLV_v11', 'PSLV_v12', 'PSLV_v13',
                 'PSLV_v14', 'PSLV_v15', 'PSLV_v16', 'PSLV_v17', 'PSLV_v18', 'PSLV_v19',
                 'MTPW_v2', 'MTPW_v3', 'MTPW_v4', 'MTPW_v5', 'MTPW_v6', 'MTPW_v7',
                 'MTPW_v8', 'MTPW_v9', 'MTPW_v10', 'MTPW_v11', 'MTPW_v12', 'MTPW_v13',
                 'MTPW_v14', 'MTPW_v15', 'MTPW_v16', 'MTPW_v17', 'MTPW_v18', 'MTPW_v19',
                 'MTPW_v20', 'MTPW_v21', 'MTPW_v22', 'IR00_v2', 'IR00_v3', 'IR00_v4',
                 'IR00_v5', 'IR00_v6', 'IR00_v7', 'IR00_v8', 'IR00_v9', 'IR00_v10',
                 'IR00_v11', 'IR00_v12', 'IR00_v13', 'IR00_v14', 'IR00_v15', 'IR00_v16',
                 'IR00_v17', 'IR00_v18', 'IR00_v19', 'IR00_v20', 'IR00_v21', 'CSST_t24',
                 'CD20_t24', 'CD26_t24', 'COHC_t24', 'DTL_t24', 'RSST_t24', 'U200_t24',
                 'U20C_t24', 'V20C_t24', 'E000_t24', 'EPOS_t24', 'ENEG_t24', 'EPSS_t24',
                 'ENSS_t24', 'RHLO_t24', 'RHMD_t24', 'RHHI_t24', 'Z850_t24', 'D200_t24',
                 'REFC_t24', 'PEFC_t24', 'T000_t24', 'R000_t24', 'Z000_t24', 'TLAT_t24',
                 'TLON_t24', 'TWAC_t24', 'TWXC_t24', 'G150_t24', 'G200_t24', 'G250_t24',
                 'V000_t24', 'V850_t24', 'V500_t24', 'V300_t24', 'TGRD_t24', 'TADV_t24',
                 'PENC_t24', 'SHDC_t24', 'SDDC_t24', 'SHGC_t24', 'DIVC_t24', 'T150_t24',
                 'T200_t24', 'T250_t24', 'SHRD_t24', 'SHTD_t24', 'SHRS_t24', 'SHTS_t24',
                 'SHRG_t24', 'PENV_t24', 'VMPI_t24', 'VVAV_t24', 'VMFX_t24', 'VVAC_t24',
                 'HE07_t24', 'HE05_t24', 'O500_t24', 'O700_t24', 'CFLX_t24', 'DELV-12']

def load_hand_data_cv():
    # Load train data
    train_df = pd.read_csv(osp.join(data_root, 'train_global_fill_REA_na_wo_img_scaled.csv'))
    train_df = train_df.loc[~((train_df.basin=='AL') & (train_df.year==2017))]
    ids = train_df['name'].values
    x_train = np.array(train_df[hand_features].values)
    y_train = train_df[['dvs24']].values

    # Load test data
    test_df = pd.read_csv(osp.join(data_root, 'train_global_fill_REA_na_wo_img_scaled.csv'))
    test_df = test_df.loc[((test_df.year==2017) & (test_df.type=='opr'))]
    x_test = np.array(test_df[hand_features].values)
    y_test = test_df[['dvs24']].values
    
    return x_train, x_test, y_train, y_test, ids

def check_stationarity(y):
    result = adfuller(y)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    return result[1] <= 0.05  # Return True if the series is stationary

def fit_arima_model(y_train, order=(1, 1, 1)):
    # Fit the ARIMA model
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    return model_fit

def forecast(model_fit, steps=10):
    # Forecast future values
    forecasted_values = model_fit.forecast(steps=steps)
    return forecasted_values

def evaluate_model(y_test, forecasted_values):
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, forecasted_values)
    print(f'Mean Squared Error: {mse}')

def plot_forecast(y_test, forecasted_values):
    # Plot the actual vs forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(forecasted_values, label='Forecasted', color='red', linestyle='--')
    plt.title('ARIMA Forecast vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

# Main execution flow
if __name__ == '__main__':
    x_train, x_test, y_train, y_test, ids = load_hand_data_cv()
    
    # Reshape y_train to be a 1D array
    y_train = y_train.flatten()
    
    # Check stationarity
    if not check_stationarity(y_train):
        # Apply differencing to make the data stationary
        y_train_diff = np.diff(y_train)
        # Check stationarity again
        assert check_stationarity(y_train_diff), "Differenced data is still non-stationary."

        # Fit ARIMA model on the differenced data
        model_fit = fit_arima_model(y_train_diff)
        
        # Forecast on the differenced data
        steps = len(y_test)  # Forecast for the length of the test set
        forecasted_values_diff = forecast(model_fit, steps=steps)

        # Inverse differencing
        forecasted_values = np.concatenate(([y_train[-1]], forecasted_values_diff)).cumsum()[1:]

    else:
        # Fit ARIMA model directly if data is already stationary
        model_fit = fit_arima_model(y_train)
        
        # Forecast values
        steps = len(y_test)  # Forecast for the length of the test set
        forecasted_values = forecast(model_fit, steps=steps)
    
    # Evaluate the model
    evaluate_model(y_test, forecasted_values)
    
    # Plot the results
    plot_forecast(y_test, forecasted_values)

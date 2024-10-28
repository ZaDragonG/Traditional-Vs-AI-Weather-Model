import numpy as np
import utils
import matplotlib.pyplot as plt
import sys
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

method = sys.argv[1]

def plot_results(y_test, y_predict, method_name):
    """
    Plot the actual vs predicted values as a dot plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, label='Actual Values', color='blue', marker='o')
    plt.scatter(range(len(y_predict)), y_predict, label=f'Predicted Values ({method_name})', color='red', marker='x')
    plt.title(f'{method_name} Predictions vs Actual Values')
    plt.xlabel('Instance')
    plt.ylabel('Tropical Cyclone Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_and_print_metrics(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

def predict_mean():
    print('Loading data...')
    _, _, y_train, y_test, _ = utils.load_hand_data_cv()
    y_predict = np.full_like(y_test, np.mean(y_train))
    compute_and_print_metrics(y_test, y_predict)
    plot_results(y_test, y_predict, 'predict_mean')

def no_change():
    print('Loading data...')
    _, _, _, y_test, _ = utils.load_hand_data_cv()
    y_predict = [0] * len(y_test)
    compute_and_print_metrics(y_test, y_predict)
    plot_results(y_test, y_predict, 'no_change')

def linear():
    from sklearn.linear_model import LinearRegression
    print('Loading data...')
    x_train, x_test, y_train, y_test, _ = utils.load_hand_data_cv()
    model = LinearRegression()
    print('Fitting model...')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    compute_and_print_metrics(y_test, y_predict)
    plot_results(y_test, y_predict, 'linear')

def lasso():
    from sklearn.linear_model import LassoCV
    print('Loading data...')
    x_train, x_test, y_train, y_test, _ = utils.load_hand_data_cv()
    y_train = y_train.ravel()
    model = LassoCV(cv=5, random_state=0, max_iter=2000)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    compute_and_print_metrics(y_test, y_predict)
    plot_results(y_test, y_predict, 'lasso')

def random_forest():
    from sklearn.ensemble import RandomForestRegressor
    print('Loading data...')
    x_train, x_test, y_train, y_test, _ = utils.load_hand_data_cv()
    y_train = y_train.ravel()
    model = RandomForestRegressor(max_depth=200, random_state=0, n_estimators=100, n_jobs=3, verbose=1)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    compute_and_print_metrics(y_test, y_predict)
    plot_results(y_test, y_predict, 'random_forest')

# Run the selected baseline method
locals()[method]()

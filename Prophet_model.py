import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv('processed_data.csv')  


data.rename(columns={'ISO_time': 'ds', 'Intensity': 'y'}, inplace=True)


data['ds'] = pd.to_datetime(data['ds'])


model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.5)
model.add_seasonality(name='yearly', period=365.25, fourier_order=10)


model.fit(data)


future = model.make_future_dataframe(periods=30)


forecast = model.predict(future)


predictions = forecast[forecast['ds'].isin(data['ds'])]
merged = pd.merge(data, predictions[['ds', 'yhat']], on='ds', how='left')


mae = np.abs(merged['y'] - merged['yhat']).mean()
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
r2 = r2_score(merged['y'], merged['yhat'])


print(f"Overall Mean Absolute Error (MAE): {mae:.2f}")
print(f"Overall Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Overall R² Score: {r2:.2f}")

merged['year'] = merged['ds'].dt.year
yearly_rmse = merged.groupby('year').apply(lambda x: np.sqrt(mean_squared_error(x['y'], x['yhat'])))

plt.figure(figsize=(12, 6))
plt.plot(yearly_rmse.index, yearly_rmse.values, marker='o', linestyle='-', color='b', markersize=8)
plt.title('Average RMSE per Year')
plt.xlabel('Year')
plt.ylabel('Average RMSE')


plt.xticks(yearly_rmse.index)  


plt.xlim(yearly_rmse.index.min() - 1, yearly_rmse.index.max() + 1) 

plt.grid()


plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True)) 

plt.show()
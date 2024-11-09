import pandas as pd
import numpy as np
 
from statsmodels.tsa.arima.model import ARIMA


data = {
    "season": ["spring", "spring", "spring", "spring", "spring", "spring", "spring", "spring", "spring", "spring", "spring", "spring"],
    "holiday": ["No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No"],
    "workingday": ["No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No"],
    "weather": ["Clear or pi", "Clear or p", "Clear or pi", "Clear or pi", "Clear or pi", "Mist", "Clear or partly cloud", "Clear or pi", "Clear or pi", "Clear or pi", "Clear or p", "Clear or pi"],
    "temp": [9.84, 9.02, 9.02, 9.84, 9.84, 9.84, 13.635, 8.2, 9.84, 13.12, 15.58, 14.76],
    "temp_feel": [14.395, 13.635, 13.635, 14.395, 14.395, 12.88, 80, 12.88, 14.395, 17.425, 19.695, 16.665],
    "humidity": [81, 80, 80, 75, 75, 75, 80, 86, 75, 76, 76, 81],
    "windspeed": [0, 0, 0, 0, 0, 6.0032, 0, 0, 0, 0, 16.9979, 19.0012],
    "demand": [2.772589, 3.688879, 3.465736, 2.564949, 0, 0, 0.693147, 1.098612, 2.079442, 2.639057, 3.583519, 4.025352]
}


#df = pd.DataFrame(data)
#df.set_index('temp', inplace=True)
#temp = data['temp']
#plt.plot(temp, label='temp')

data = pd.read_csv('TestCSVData.csv')
time_index = np.arange(len(data))  #create time series indices
"""
plt.plot(time_index, data['temp'])  #convert the temp data into a time series
#plt.plot(data)
plt.xlabel('Temp')
plt.ylabel('Value')
plt.title('Temp data values')
plt.show()
"""

from statsmodels.tsa.stattools import adfuller
#print('reached')

data['temp'] = data['temp'].fillna(0)     # replace missing value with 0
result = adfuller(data['temp'])
print('ADF stats : ', result[0])
print('p-value : ', result[1])

# apply data differencing for non statinary value
diff_data = data['temp'].diff().dropna()
print(diff_data)

"""
# checking for non stationarity after first differencing
result = adfuller(diff_data)
print('ADF stats : ', result[0])
print('p-value : ', result[1])
"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(diff_data)
plot_pacf(diff_data)
plt.show()

model = ARIMA(diff_data, order=(1,1,1))
result = model.fit()

# forecasting
forecast = result.predict(start=len(data['temp']), end=len(data['temp']) + 20, typ='levels')

# visualize forecast
plt.plot(data['temp'], label='original data')
plt.plot(forecast, label='forecast')
plt.xlabel('date')
plt.ylabel('value')
plt.title('Arima forecast')
plt.legend()
plt.show()
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prettytable import PrettyTable

# populate the dataframe 'data'
data = pd.read_csv("CarSharing.csv")

data.dropna(subset=['demand'], inplace=True)     # drop only rows that have missing demand value

data["timestamp"] = pd.to_datetime(data["timestamp"])    # Convert timestamps to datetime

weekly_data = data.resample("W-SUN", on="timestamp")["demand"].mean()   # Resample data to get weekly average demand
weekly_data.rename("Weekly Average Demand", inplace=True)
weekly_data.index.freq = 'W-SUN'

# split the data (train-test)
split_point = int(len(weekly_data) * 0.7)

# train_data = weekly_data[:split_point]   
# test_data = weekly_data[split_point:]   

# Create DataFrame with datetime index and set frequency to 'W-SUN'
train_data = pd.DataFrame({"demand": weekly_data[:split_point]}, index=pd.date_range(start=weekly_data.index[0], periods=len(weekly_data[:split_point]), freq='W-SUN'))
test_data = pd.DataFrame({"demand": weekly_data[split_point:]}, index=pd.date_range(start=weekly_data.index[0], periods=len(weekly_data[:split_point]), freq='W-SUN'))


table = PrettyTable()
table.field_names = ["ADF Statistic", "P-value", "d"]

# Perform differencing if the ADF test hints at non-stationarity
def check_stationarity(data):
    # Handle missing values
    data = data.dropna()
    
    result = adfuller(data)
    adf_stats, p_value = result[0], result[1]
    # Add data to the table
    table.add_row([adf_stats, p_value, d])
    print(table)

    if p_value <= 0.05:
        print("Series is stationary at significance level 0.05 at d = ", d)
        return True
    else:
        print("Series is not stationary at significance level 0.05. Performing differencing ...\n")
        return False

max_differencing = 2  # Set a maximum level of differencing
d = 0       
is_stationary = check_stationarity(train_data)

while not is_stationary:
    if d >= max_differencing:
        print("Stationarity not achieved after maximum differencing.")
        break
    train_data = train_data.diff().dropna()
    d += 1
    is_stationary = check_stationarity(train_data)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plot_acf(train_data)
plot_pacf(train_data)
plt.show()

p = int(input("Enter the value of p : "))
q = int(input("Enter the value of q : "))


train_data.index.freq = 'W-SUN'
test_data.index.freq = 'W-SUN'

# Fit ARIMA model
model = ARIMA(train_data, order=(1, 1, 1))  # d = 1 as obtained before
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test_data))


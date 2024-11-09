import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# populate the dataframe 'data'
data = pd.read_csv("CarSharing.csv")

data.dropna(subset=['demand'], inplace=True)     # drop only rows that has missing demand value

data["timestamp"] = pd.to_datetime(data["timestamp"])    # Convert timestamps to datetime

weekly_data = data.resample("W-SUN", on="timestamp")["demand"].mean()   # Resample data to get weekly average demand

# split the data (train-test)
split_point = int(len(weekly_data) * 0.7)

train_data = weekly_data[:split_point]   
test_data = weekly_data[split_point:]   

# ARIMA models require stationary data. Check for stationarity by plotting the time series and its differenced version
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['demand'])

# result[0] -> ADF Statistic: A more negative ADF statistic suggests stronger evidence in favor of stationarity.
# result[1] -> p-value: A small p-value (p < 0.05) suggests that the ADF statistic is statistically significant, providing evidence to reject the null hypothesis of non-stationarity.
# result[4] contains critical values at different significance levels (1%, 5%, 10%)

from prettytable import PrettyTable

table = PrettyTable()
table.field_names = ["ADF Statistic", "P-value", "Critical Values"]

# Add data to the table
table.add_row([result[0], result[1], ", ".join(f"{key}: {value:.4f}" for key, value in result[4].items())])

print(table)

# Perform differencing if the ADF test hints at non-stationarity
def check_stationarity(data, significance_level=0.05):
    adf_result = adfuller(data)
    statistic, p_value, critical_values = adf_result[0], adf_result[1], adf_result[4]

    if p_value <= significance_level:
        print("Series is stationary at significance level", significance_level)
        return True
    else:
        for name, value in critical_values.items():
            if p_value > value:
                print(f"Series is not stationary at significance level {significance_level} (critical value for {name}: {value:.4f})")
                return False
        return False  # In case critical values aren't explicitly listed

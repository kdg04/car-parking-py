import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

# Load the CSV file
df = pd.read_csv('Updated_CarSharing.csv')

# Filter data for the year 2017 
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_2017 = df[df['timestamp'].dt.year == 2017]

df_2017 = df_2017.set_index('timestamp')
df_2017 = df_2017.asfreq('D')  # Set frequency to daily

# Plot time series for each column
fig, axs = plt.subplots(4, 1, figsize=(10, 12))
axs[0].plot(df_2017['temp'], label='Temperature')
axs[0].set_ylabel('Temperature')
axs[1].plot(df_2017['humidity'], label='Humidity')
axs[1].set_ylabel('Humidity')
axs[2].plot(df_2017['windspeed'], label='Windspeed')
axs[2].set_ylabel('Windspeed')
axs[3].plot(df_2017['demand'], label='Demand')
axs[3].set_ylabel('Demand')

for ax in axs:
    ax.legend()
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# Seasonal decomposition with manual seasonal period specification
temp_stl = STL(df_2017['temp'], seasonal=31)  # Assuming weekly seasonality
temp_res = temp_stl.fit()
temp_res.plot()
plt.show()

# Autocorrelation analysis for demand
plot_acf(df_2017['demand'], lags=50)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Demand')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame (replace this with your actual DataFrame)
data = {
    'temp': [25.0, 20.0, 22.0, 18.0, 30.0],
    'humidity': [50.0, 45.0, 55.0, 60.0, 40.0],
    'windspeed': [3.5, 4.0, 3.2, 2.8, 4.5],
    'demand': [25, 30, 20, 15, 35],
    'demand_category': [1, 1, 2, 2, 1]
}
df = pd.DataFrame(data)

# Create a table and display it
table = plt.table(cellText=df.values, colLabels=df.columns, loc='center')


# Hide axes
plt.axis('off')

# Show the plot
plt.show()
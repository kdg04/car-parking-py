import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('preprocessed_CarSharing.csv')

average_demand = df['demand'].mean()                     # calculate the average demand rate

# create a new column for the labels (1 for above average, 2 for below average)
df['label'] = df['demand'].apply(lambda x: 1 if x > average_demand else 2)

# Split data into training and testing sets (30% for testing)
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.3, random_state=10)

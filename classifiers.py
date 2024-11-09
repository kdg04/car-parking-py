import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('CarSharing.csv')

# Extract the demand column and drop rows with NaN values
df = df[['temp', 'humidity', 'windspeed', 'demand']].dropna()

# Calculate the average demand rate
average_demand_rate = df['demand'].mean()

# Categorize the demand rate into two groups based on the average
df['demand_category'] = df['demand'].apply(lambda x: 1 if x > average_demand_rate else 2)

# Split the data into training and testing sets (70% training, 30% testing)
X = df[['temp', 'humidity', 'windspeed', 'demand']]  # Features
y = df['demand_category']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the labels using the testing data
y_pred = model.predict(X_test)

# Plot the demand values against their labels
plt.figure(figsize=(10, 6))
plt.scatter(df['demand'], df['demand_category'], c='b', marker='o', label='Actual Labels')
plt.scatter(X_test['demand'], y_pred, c='r', marker='x', label='Predicted Labels')
plt.xlabel('Demand')
plt.ylabel('Label')
plt.title('Demand vs. Labels')
plt.legend()
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of logistic regression model: {accuracy:.2f}")

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2'], yticklabels=['1', '2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



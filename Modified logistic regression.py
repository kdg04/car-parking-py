import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load your data (assuming you have already loaded and preprocessed it)
data = pd.read_csv('Updated_CarSharing.csv')

# Calculate average demand rate
average_demand_rate = data['demand'].mean()

"""
# Categorize demand rates into two groups based on average
data['label'] = np.where(data['demand'] > average_demand_rate, 1, 2)

# Split data into train and test sets
X = data.drop(['demand', 'label'], axis=1)  # Assuming 'demand' column is the target variable
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
"""

# Categorize the demand rate into two groups based on the average
data['label'] = data['demand'].apply(lambda x: 1 if x > average_demand_rate else 2)

# Split the data into features (X) and target variable (y)
X = data[['temp', 'humidity', 'windspeed']]  # Features
y = data['label']  # Target variable

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Predict labels for the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Plot demand values against their predicted labels
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Demand Values vs Predicted Labels')
plt.grid(True)
plt.show()

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

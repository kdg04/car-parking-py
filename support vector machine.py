import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('Updated_CarSharing.csv')

# Replace NaN values with mean of the column
imputer = SimpleImputer(strategy='mean')
df[['temp', 'humidity', 'windspeed']] = imputer.fit_transform(df[['temp', 'humidity', 'windspeed']])

# Calculate the average demand rate
average_demand_rate = df['demand'].mean()

# Categorize the demand rate into two groups based on the average
df['demand_category'] = df['demand'].apply(lambda x: 1 if x > average_demand_rate else 2)

# Split the data into features (X) and target variable (y)
X = df[['temp', 'humidity', 'windspeed']]  # Features
y = df['demand_category']  # Target variable

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = svm_classifier.predict(X_train)

# Make predictions on the testing data
y_test_pred = svm_classifier.predict(X_test)

# Calculate the accuracy of the model on training and testing data
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on training set: {train_accuracy:.2f}")
print(f"Accuracy on test set: {test_accuracy:.2f}")

# Plot the training set results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_train['temp'], X_train['humidity'], c=y_train, cmap='viridis')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Training Set Results')

# Plot the test set results
plt.subplot(1, 2, 2)
plt.scatter(X_test['temp'], X_test['humidity'], c=y_test, cmap='viridis')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Test Set Results')

plt.tight_layout()
plt.show()

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2'], yticklabels=['1', '2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.show()

import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr

# Load data (replace with your data loading method)
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

df = pd.DataFrame(data)

# Define significance level
alpha = 0.05

# Test for categorical variables (season, weather)
for col in ["season", "weather"]:
    contingency_table = pd.crosstab(df[col], df["demand"])
    chi2, pval, *_ = chi2_contingency(contingency_table.values)
    result = "reject" if pval < alpha else "fail to reject"
    print(f"{col}: Chi-square test - H0: No relationship, result: {result}, p-value: {pval:.4f}")

# Test for continuous variables (temp, temp_feel, humidity, windspeed)
for col in ["temp", "temp_feel", "humidity", "windspeed"]:
    corr, pval = pearsonr(df[col], df["demand"])
    result = "reject" if pval < alpha else "fail to reject"
    print(f"{col}: Pearson correlation - H0: No linear relationship, result: {result}, p-value: {pval:.4f}")

# Alternative test for non-linear relationships (optional)
for col in ["temp", "temp_feel", "humidity", "windspeed"]:
    corr, pval = spearmanr(df[col], df["demand"])
    result = "reject" if pval < alpha else "fail to reject"
    print(f"{col}: Spearman correlation (non-linear) - H0: No relationship, result: {result}, p-value: {pval:.4f}")

# Note: This code assumes the data is preprocessed and cleaned.

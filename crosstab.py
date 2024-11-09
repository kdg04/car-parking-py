import pandas as pd

# Create a sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C'],
    'Value1': [10, 15, 12, 8, 11, 9, 14, 13, 10],
    'Value2': [25, 30, 27, 20, 22, 21, 26, 24, 23]
}

df = pd.DataFrame(data)

# Compute a simple crosstab
crosstab = pd.crosstab(df['Category'], [df['Value1'], df['Value2']])

print(crosstab)

# Compute a crosstab with two variables
#crosstab = pd.crosstab(df['Category'], df['Value'])
#print(crosstab)

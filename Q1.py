import pandas as pd

df = pd.read_csv('CarSharing.csv')       # load the file into dataframe

df.drop_duplicates(inplace=True)          # Drop duplicate rows

# Function to handle null values
def handle_null_values(column):
    if column == 'id':
        df.dropna(subset=[column], inplace=True) 
        return
    if column == 'timestamp':
        df['timestamp'] = df['timestamp'].ffill()
        return
    if df[column].dtype == 'object':             # Categorical data (strings)      
        df[column] = df[column].fillna(df[column].mode().iloc[0])      # replace with the most frequent value
    else:                                        # Numeric data
       if df[column].sum() > 0:
            df[column] = df[column].fillna(df[column].mean())

for col in df.columns:    # for each column of the file
    handle_null_values(col)

df.to_csv('Updated_CarSharing.csv', index=False)   # Save the preprocessed data to a new CSV file

print("New File Updated_CarSharing saved")

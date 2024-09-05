import pandas as pd

# Load the CSV data into a DataFrame
data = pd.read_csv('data/combined_data.csv')

# Convert Page Views and Previous Visits to integers
data['Page Views'] = data['Page Views'].astype(int)
data['Previous Visits'] = data['Previous Visits'].astype(int)

# Convert other fields to float with 14 decimal places
float_columns = ['Session Duration', 'Time on Page']
for col in float_columns:
    data[col] = data[col].astype(float).round(14)

# Multiply Session Duration by 60
data['Session Duration'] = data['Session Duration'] * 60

# Remove records where Session Duration < Time on Page
data = data[data['Session Duration'] >= data['Time on Page']]

# Save the cleaned data to a new CSV file
data.to_csv('data/cleaned_data.csv', index=False)

print("Data cleaned and saved to 'data/cleaned_data.csv'")

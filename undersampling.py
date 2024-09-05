import pandas as pd

# Step 1: Load the data
data = pd.read_csv('data/cleaned_data.csv')

# Step 2: Separate the 'Bot' and 'Human' rows
human_data = data[data['Visitor Type'] == 'Human']
bot_data = data[data['Visitor Type'] == 'Bot']

# Step 3: Check if the number of 'Human' rows is greater than 'Bot' rows
# If so, sample with replacement to avoid the error
if len(human_data) > len(bot_data):
    bot_data_undersampled = bot_data.sample(n=len(human_data), replace=True, random_state=42)
else:
    bot_data_undersampled = bot_data.sample(n=len(human_data), random_state=42)

# Step 4: Combine the undersampled 'Bot' data with 'Human' data
balanced_data = pd.concat([human_data, bot_data_undersampled])

# Step 5: Shuffle the balanced dataset to mix 'Bot' and 'Human' examples
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv('data/balanced_data.csv', index=False)

print("Balanced data saved to 'data/balanced_data.csv'.")

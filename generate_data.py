import numpy as np
import pandas as pd

# Step 1: Load Human Data from CSV
human_df = pd.read_csv('data/human_data.csv')

# Add a label to distinguish human data
human_df['Visitor Type'] = 'Human'

# Step 2: Generate Synthetic Bot Data with specified characteristics
np.random.seed(42)  # Seed for reproducibility
num_bots = 1000

# Parameters for distributions
page_views_mean_bot = 30
session_duration_scale_bot = 2
time_on_page_mean_bot = 3
previous_visits_lambda_bot = 1

# Generate synthetic bot data with the required characteristics and noise
bot_data = {
    "Page Views": np.random.normal(loc=page_views_mean_bot, scale=5, size=num_bots).clip(1, 40),
    
    "Session Duration": np.random.exponential(scale=session_duration_scale_bot, size=num_bots).clip(0, 20),
    
    "Time on Page": np.random.lognormal(mean=np.log(time_on_page_mean_bot), sigma=0.5, size=num_bots).clip(0, 30),
    
    "Previous Visits": np.random.poisson(lam=previous_visits_lambda_bot, size=num_bots).clip(0, 10)
}

# Convert to DataFrame
bot_df = pd.DataFrame(bot_data)

# Add a label to distinguish bot data
bot_df['Visitor Type'] = 'Bot'

# Step 3: Combine and Visualize Data
# Combine the human and bot datasets for comparison
combined_df = pd.concat([human_df, bot_df])

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save the combined dataset to a CSV file
combined_df.to_csv('data/combined_data.csv', index=False)

print("Data generated and saved to 'data/combined_data.csv'")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('data/balanced_data.csv')

# Separate the data into humans and bots
df_human = df[df['Visitor Type'] == 'Human']
df_bot = df[df['Visitor Type'] == 'Bot']

# Define a function to create and save graphs
def plot_and_save(df1, df2, column, filename):
    plt.figure(figsize=(12, 6))
    
    sns.histplot(df1[column], color='blue', label='Human', kde=True, bins=30)
    sns.histplot(df2[column], color='red', label='Bot', kde=True, bins=30)
    
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Set style
sns.set_style(style="whitegrid")

# Plot and save distributions of Page Views
plt.figure(figsize=(12, 6))
sns.histplot(df, x="Page Views", hue="Visitor Type", element="step", stat="density", common_norm=False)
plt.title('Distribution of Page Views')
plt.savefig('graphs/page_views_distribution.png')  # Save the figure to a file
plt.close()  # Close the plot to free memory

# Plot and save distributions of Session Duration
plt.figure(figsize=(12, 6))
sns.histplot(df, x="Session Duration", hue="Visitor Type", element="step", stat="density", common_norm=False)
plt.title('Distribution of Session Duration')
plt.savefig('graphs/session_duration_distribution.png')
plt.close()

# Plot and save distributions of Time on Page
plt.figure(figsize=(12, 6))
sns.histplot(df, x="Time on Page", hue="Visitor Type", element="step", stat="density", common_norm=False)
plt.title('Distribution of Time on Page')
plt.savefig('graphs/time_on_page_distribution.png')
plt.close()

# Plot and save box plot comparison of Page Views
plt.figure(figsize=(14, 8))
sns.boxplot(x="Visitor Type", y="Page Views", data=df)
plt.title('Box Plot of Page Views')
plt.savefig('graphs/box_plot_page_views.png')
plt.close()

# Plot and save box plot comparison of Session Duration
plt.figure(figsize=(14, 8))
sns.boxplot(x="Visitor Type", y="Session Duration", data=df)
plt.title('Box Plot of Session Duration')
plt.savefig('graphs/box_plot_session_duration.png')
plt.close()

# Plot and save box plot comparison of Time on Page
plt.figure(figsize=(14, 8))
sns.boxplot(x="Visitor Type", y="Time on Page", data=df)
plt.title('Box Plot of Time on Page')
plt.savefig('graphs/box_plot_time_on_page.png')
plt.close()

print("Graphs have been saved successfully.")

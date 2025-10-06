#a place for cow data



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io # To simulate reading from a file using the provided text

# --- Option 1: Load from a CSV file (Recommended for larger datasets) ---
# Assuming your data looks like the image and is saved as 'sensor_data.csv'
# Make sure the CSV doesn't have the units row, just the header and data.
# Example CSV content:
# Temperature (°F),Humidity (%),CO2 (ppm),NH3 (ppm),H2S (ppm),CH4 (ppm),Timestamp
# 88.29,47.17,406,3.50862,0.960586,1.5712,2025-04-17 13:13:53
# 88.36,47.17,406,3.50587,0.961188,1.57198,2025-04-17 13:13:48
# ... and so on

# try:
#     df = pd.read_csv('sensor_data.csv')
# except FileNotFoundError:
#     print("Error: 'sensor_data.csv' not found. Using sample data instead.")
#     # Fallback to sample data if file isn't present

# --- Option 2: Use the sample data directly from the image (for demonstration) ---
# Manually transcribed data from the image

# Use io.StringIO to simulate reading the string data as a file
#df = pd.read_csv(io.StringIO(data))

df = pd.read_csv('cow_it.csv')

# --- Data Cleaning and Preparation ---

# 1. Convert Timestamp column to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 2. Sort by Timestamp (important for time-series analysis)
# Your sample data is newest-first, let's make it chronological
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# 3. Set Timestamp as the index (useful for time-based operations)
df = df.set_index('Timestamp')

# 4. Optional: Rename columns for easier access (remove special chars/units)
df.columns = ['Temperature_F', 'Humidity_pct', 'CO2_ppm', 'NH3_ppm', 'H2S_ppm', 'CH4_ppm']

# Display the first few rows and info to verify
print("--- Cleaned Dataframe Head ---")
print(df.head())
print("\n--- Dataframe Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
# .T transposes the output for potentially better readability
print(df.describe().T)


fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(12, 15), sharex=True)
fig.suptitle('Sensor Readings Over Time', fontsize=16)

for i, col in enumerate(df.columns):
    axes[i].plot(df.index, df[col], label=col)
    axes[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F')) # Nicer labels
    axes[i].legend(loc='upper left')
    axes[i].grid(True)

plt.xlabel('Timestamp')
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
plt.savefig('cow_basic.png')


# Calculate the correlation matrix
correlation_matrix = df.corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix)

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Sensor Readings')
plt.savefig('cow_corrlelation.png')

# Resample to 1-minute averages
df_resampled_minute = df.resample('1T').mean() # '1T' means 1 minute frequency, use mean aggregation

print("\n--- Data Resampled to 1-Minute Averages ---")
print(df_resampled_minute.head())

# You can plot the resampled data similarly to the original data
# Example for Temperature:
plt.figure(figsize=(12, 4))
plt.plot(df_resampled_minute.index, df_resampled_minute['Temperature_F'], label='1-Min Avg Temp')
plt.plot(df.index, df['Temperature_F'], label='Original Temp (5s)', alpha=0.5) # Plot original for comparison
plt.title('Temperature (Original vs. 1-Minute Average)')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.grid(True)
plt.savefig('cow_resample_to_1_min.png')
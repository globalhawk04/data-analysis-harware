# --- Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Added for potential advanced calculations
from statsmodels.tsa.seasonal import seasonal_decompose # Added for decomposition

# --- Load Data ---
try:
    # Best practice: Specify parse_dates directly during loading
    df = pd.read_csv('sensor_readings_export.csv', parse_dates=['Timestamp'])
    print("Successfully loaded 'sensor_readings_export.csv'")
except FileNotFoundError:
    print("Error: 'sensor_readings_export.csv' not found. Please ensure the file is in the correct directory.")
    # In a real scenario, you might exit or use fallback sample data here.
    # For this example, we'll assume the file exists.
    exit() # Exit if file not found for this analysis

# --- Data Cleaning and Preparation ---

# 1. Timestamp already parsed during load
# 2. Sort by Timestamp (ensure chronological order)
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# 3. Set Timestamp as the index
df = df.set_index('Timestamp')

# 4. Rename columns for easier access
df.columns = ['Temperature_F', 'Humidity_pct', 'CO2_ppm',
              'NH3_ppm', 'H2S_ppm', 'CH4_ppm']

# 5. Optional: Check for Missing Values (Good Practice)
print("\n--- Missing Value Check ---")
print(df.isnull().sum())
# Consider imputation strategies (e.g., ffill, bfill, interpolation) if NaNs are present

# --- Initial Data Overview ---
print("\n--- Cleaned Dataframe Head ---")
print(df.head())
print("\n--- Dataframe Info ---")
df.info()
print("\n--- Descriptive Statistics ---")
# .T transposes the output for potentially better readability
print(df.describe().T)

# --- Distribution Analysis ---
print("\n--- Analyzing Sensor Distributions ---")
n_cols = len(df.columns)
n_rows = (n_cols + 1) // 2 # Arrange plots in two columns

fig_dist, axes_dist = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, n_rows * 4))
axes_dist = axes_dist.flatten() # Flatten the axes array for easy iteration

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, ax=axes_dist[i])
    axes_dist[i].set_title(f'Distribution of {col}')
    axes_dist[i].set_xlabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
    axes_dist[i].set_ylabel('Frequency')

# Hide any unused subplots if n_cols is odd
if n_cols % 2 != 0:
    axes_dist[-1].set_visible(False)

plt.tight_layout()
plt.savefig('cow_distributions.png')
print("Saved sensor distribution plots to 'cow_distributions.png'")
# plt.show() # Optional: display plots interactively

# --- Box Plots for Outlier Visualization ---
fig_box, axes_box = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, n_rows * 4))
axes_box = axes_box.flatten()

for i, col in enumerate(df.columns):
    sns.boxplot(x=df[col], ax=axes_box[i])
    axes_box[i].set_title(f'Box Plot of {col}')
    axes_box[i].set_xlabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))

if n_cols % 2 != 0:
    axes_box[-1].set_visible(False)

plt.tight_layout()
plt.savefig('cow_boxplots.png')
print("Saved sensor box plots to 'cow_boxplots.png'")


# --- Time Series Plotting (Original) ---
print("\n--- Plotting Raw Sensor Data Over Time ---")
fig_ts, axes_ts = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(15, 18), sharex=True)
fig_ts.suptitle('Sensor Readings Over Time (Raw Data)', fontsize=18, y=1.02) # Adjust y for title position

for i, col in enumerate(df.columns):
    axes_ts[i].plot(df.index, df[col], label=col, linewidth=1) # Thinner line for dense data
    axes_ts[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
    # Add horizontal lines for mean or specific thresholds if known/relevant
    # mean_val = df[col].mean()
    # axes_ts[i].axhline(mean_val, color='r', linestyle='--', linewidth=0.8, label=f'Mean: {mean_val:.2f}')
    axes_ts[i].legend(loc='upper left')
    axes_ts[i].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xlabel('Timestamp')
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
plt.savefig('cow_timeseries_raw.png')
print("Saved raw time series plots to 'cow_timeseries_raw.png'")


      
# --- Correlation Analysis ---
print("\n--- Calculating and Visualizing Correlation Matrix ---")
correlation_matrix = df.corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix)

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Sensor Readings', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('cow_correlation_heatmap.png')
print("Saved correlation heatmap to 'cow_correlation_heatmap.png'")
# plt.show()

# --- Pair Plot for Detailed Relationships ---
# Creates scatter plots for joint relationships and histograms for univariate distributions
print("\n--- Generating Pair Plot (takes a moment) ---")
# Reduce sample size if dataset is very large to speed up pairplot generation
# df_sample = df.sample(n=1000) if len(df) > 1000 else df
# sns.pairplot(df_sample) # Use df_sample instead of df if needed

pair_plot = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha':0.6, 's':10}, height=2) # Use kde for diagonal, adjust point size/alpha
pair_plot.fig.suptitle('Pairwise Relationships Between Sensor Readings', y=1.02, fontsize=16)
plt.savefig('cow_pairplot.png')
print("Saved pair plot to 'cow_pairplot.png'")
# plt.show()

      
# --- Resampling Data ---
# Resample to 1-minute averages (adjust frequency 'T' as needed: '5T', '15T', 'H')
resample_freq = '1T'
df_resampled = df.resample(resample_freq).mean() # Use mean, median, min, max, etc. as appropriate

print(f"\n--- Data Resampled to {resample_freq} Averages ---")
print(df_resampled.head())

# --- Plotting Resampled Data ---
print(f"\n--- Plotting Resampled ({resample_freq}) Sensor Data Over Time ---")
fig_res, axes_res = plt.subplots(nrows=len(df_resampled.columns), ncols=1, figsize=(15, 18), sharex=True)
fig_res.suptitle(f'Sensor Readings Over Time ({resample_freq} Average)', fontsize=18, y=1.02)

for i, col in enumerate(df_resampled.columns):
    axes_res[i].plot(df_resampled.index, df_resampled[col], label=f'{col} ({resample_freq} Avg)', linewidth=1.5)
    # Optionally plot original data lightly in the background for comparison
    # axes_res[i].plot(df.index, df[col], label=f'{col} (Raw)', alpha=0.3, linewidth=0.5)
    axes_res[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
    axes_res[i].legend(loc='upper left')
    axes_res[i].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xlabel('Timestamp')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig(f'cow_timeseries_resampled_{resample_freq}.png')
print(f"Saved resampled ({resample_freq}) time series plots to 'cow_timeseries_resampled_{resample_freq}.png'")
# plt.show()

# --- Comparison Plot (Example: Temperature) ---
plt.figure(figsize=(15, 5))
plt.plot(df_resampled.index, df_resampled['Temperature_F'], label=f'Temp ({resample_freq} Avg)', color='red', linewidth=2)
plt.plot(df.index, df['Temperature_F'], label='Temp (Raw)', alpha=0.4, color='blue', linewidth=0.7)
plt.title(f'Temperature Comparison: Raw vs. {resample_freq} Average')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'cow_temp_resample_comparison_{resample_freq}.png')
print(f"Saved temperature comparison plot to 'cow_temp_resample_comparison_{resample_freq}.png'")
# plt.show()

      
# --- Rolling Statistics ---
# Define window size (e.g., 12 samples = 1 minute for 5s data, 60 samples = 5 minutes)
window_size = 12 * 5 # 5 minutes of data (12 samples/min * 5 min)
window_label = '5min'

print(f"\n--- Calculating Rolling Statistics (Window: {window_label}) ---")

# Create a new figure for rolling stats
fig_roll, axes_roll = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(15, 18), sharex=True)
fig_roll.suptitle(f'Rolling Mean & Standard Deviation ({window_label} Window)', fontsize=18, y=1.02)

for i, col in enumerate(df.columns):
    rolling_mean = df[col].rolling(window=window_size).mean()
    rolling_std = df[col].rolling(window=window_size).std()

    axes_roll[i].plot(df.index, df[col], color='lightblue', alpha=0.6, label=f'{col} (Raw)', linewidth=0.8)
    axes_roll[i].plot(rolling_mean.index, rolling_mean, color='blue', label=f'Rolling Mean ({window_label})', linewidth=1.5)
    # Optionally plot std dev on the same or secondary axis
    ax2 = axes_roll[i].twinx() # Secondary y-axis for std dev
    ax2.plot(rolling_std.index, rolling_std, color='red', linestyle='--', label=f'Rolling Std Dev ({window_label})', linewidth=1)
    ax2.set_ylabel('Std Dev', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.spines['right'].set_color('red') # Color the axis spine

    # Combine legends
    lines, labels = axes_roll[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes_roll[i].legend(lines + lines2, labels + labels2, loc='upper left')

    axes_roll[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
    axes_roll[i].grid(True, which='major', linestyle='--', linewidth=0.5)


plt.xlabel('Timestamp')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig(f'cow_rolling_stats_{window_label}.png')
print(f"Saved rolling statistics plots to 'cow_rolling_stats_{window_label}.png'")
# plt.show()

# --- Time Series Decomposition (Example for Temperature) ---
# Note: Requires sufficient data and potential seasonality to be meaningful.
# 'period' should correspond to the expected seasonal cycle length in terms of data points.
# E.g., if data is 1-minute sampled and you expect daily seasonality: period = 60 * 24 = 1440
# For 5-second data and daily seasonality: period = 12 * 60 * 24 = 17280
# Adjust 'period' based on your data frequency and expected cycles. Check data length first.

# Let's assume we have enough data for a *short* cycle or just want to see trend/residual
# For demonstration, using a shorter period (e.g., 1 hour = 12*60=720 points for 5s data)
# Choose a relevant variable like Temperature
variable_to_decompose = 'Temperature_F'
decomposition_period = 12 * 60 # Example: 1 hour period for 5s data

if len(df) > 2 * decomposition_period: # Need at least 2 full periods for decomposition
    print(f"\n--- Decomposing {variable_to_decompose} Time Series (Period: {decomposition_period} samples) ---")
    try:
        # Use additive model if magnitude of seasonality doesn't depend on trend level
        # Use multiplicative if seasonality increases/decreases with the trend
        decomposition = seasonal_decompose(df[variable_to_decompose], model='additive', period=decomposition_period)

        fig_decomp = decomposition.plot()
        fig_decomp.set_size_inches(14, 10)
        fig_decomp.suptitle(f'Time Series Decomposition of {variable_to_decompose}', y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(f'cow_decomposition_{variable_to_decompose}.png')
        print(f"Saved decomposition plot to 'cow_decomposition_{variable_to_decompose}.png'")
        # plt.show()
    except ValueError as e:
        print(f"Could not perform decomposition, possibly insufficient data or period too large: {e}")
else:
    print(f"\n--- Skipping Decomposition: Insufficient data length ({len(df)}) for period {decomposition_period} ---")
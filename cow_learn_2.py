import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.dates as mdates # Import the dates module

# Load data and initial processing
df = pd.read_csv('sensor_learn_data.csv')
df = df.drop(columns=['ID'])

# Convert Timestamp to datetime *first* - this is essential!
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by timestamp (good practice)
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# Set Timestamp as the index - this creates the DatetimeIndex
df = df.set_index('Timestamp')

# print("DataFrame info after setting Timestamp as index:")
# print(df.info()) # You can uncomment this to confirm Timestamp is the index and its dtype is datetime64[ns]

# Define time ranges for baselines and test periods
# Correcting typo in fram_base_end_end
baseline1_start = "2025-04-17 12:15:04"
baseline1_end = '2025-04-17 12:26:57'

hay_start =  "2025-04-17 12:27:01"
cowpatty_start = "2025-04-17 12:42:05"
dirt_start = "2025-04-17 13:02:02"
grass_start ="2025-04-17 13:17:03"
farm_end = "2025-04-17 13:33:25"

baseline2_start = "2025-04-17 13:58:05"
baseline2_end = "2025-04-17 14:03:08" # Corrected variable name


# --- Step 1 & 2: Isolate Baseline Data and Calculate Averages ---

# Baseline 1 (Before Test)
baseline1_df = df[baseline1_start : baseline1_end]
baseline1_averages = None # Initialize to None
if baseline1_df.empty:
    print(f"Warning: No data found in Baseline 1 period ({baseline1_start} to {baseline1_end}). Cannot calculate its average.")
else:
    baseline1_averages = baseline1_df.mean()
    print("\nBaseline 1 Averages (Before Test):")
    print(baseline1_averages)
    print("-" * 40)

# Baseline 2 (After Test)
baseline2_df = df[baseline2_start : baseline2_end]
baseline2_averages = None # Initialize to None
if baseline2_df.empty:
    print(f"Warning: No data found in Baseline 2 period ({baseline2_start} to {baseline2_end}). Cannot calculate its average.")
else:
    baseline2_averages = baseline2_df.mean()
    print("\nBaseline 2 Averages (After Test):")
    print(baseline2_averages)
    print("-" * 40)


# --- Step 3: Define the data to be plotted ---
# To show both baselines and the test periods, the plotting range should cover
# from the start of the first baseline to the end of the second baseline.
plot_df = df[baseline1_start : baseline2_end]

if plot_df.empty:
    print(f"Error: No data found in the overall plotting range ({baseline1_start} to {baseline2_end}). Exiting.")
    exit()

print(f"Plotting Data Range: {baseline1_start} to {baseline2_end}")
print(f"Number of data points to plot: {len(plot_df)}")
print("-" * 40)

# --- Plotting ---
# Create the figure and subplots
fig, axes = plt.subplots(nrows=len(plot_df.columns), ncols=1, figsize=(12,20), sharex=True) # Increased figsize slightly

fig.suptitle('Sensor Readings vs. Baseline Averages', fontsize=16) # Update title

# Ensure axes is iterable even if only one column
if len(plot_df.columns) == 1:
    axes = [axes] # Make it a list if only one subplot


for i, col in enumerate(plot_df.columns):
    ax = axes[i] # Get the current axis for this column

    # Plot the time series data for the main plotting range
    ax.plot(plot_df.index, plot_df[col], label=col)

    # --- Plot the Baseline Average Lines ---
    # Plot Baseline 1 average if it was calculated and exists for this column
    if baseline1_averages is not None and col in baseline1_averages.index:
         ax.axhline(y=baseline1_averages[col], color='red', linestyle='--', label='Baseline 1 Avg')

    # Plot Baseline 2 average if it was calculated and exists for this column
    if baseline2_averages is not None and col in baseline2_averages.index:
         ax.axhline(y=baseline2_averages[col], color='blue', linestyle='-.', label='Baseline 2 Avg') # Use a different style/color


    ax.set_ylabel(col)
    ax.legend(loc='upper left')
    ax.grid(True)


# --- Configure the X-axis for date/time display ---
# Apply this to the bottom-most axis (or the single axis)
ax_bottom = axes[-1] # Get the last axis

# Set the major locator to place ticks every 1 minute
ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))

# Set the major formatter to display HH:MM:SS
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

# Optional: Set minor ticks for finer granularity (e.g., every 15 seconds)
ax_bottom.xaxis.set_minor_locator(mdates.SecondLocator(interval=15))


# Improve the layout and prevent labels overlapping
plt.xlabel('Timestamp') # Still set the overall xlabel
fig.autofmt_xdate() # Auto-format the x-axis labels (rotate them)
plt.tight_layout(rect=[0,0.03,1,0.98]) # Adjust layout to make room for suptitle and labels

# Save the figure
plt.savefig('cow_learn_baselines.png') # Changed filename
#plt.show() # Display the plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
# import matplotlib.dates as mdates # Not needed for correlation matrix heatmaps


# Load data and initial processing
df = pd.read_csv('sensor_learn_data.csv')
# Dropping only the 'ID' column as per your provided code
# This means Temperature and Humidity ARE included
df = df.drop(columns=['ID'])

# Convert Timestamp to datetime *first* - this is essential!
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by timestamp (good practice)
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# Set Timestamp as the index - this creates the DatetimeIndex
df = df.set_index('Timestamp')

# print("DataFrame info after setting Timestamp as index:")
# print(df.info()) # You can uncomment this to confirm Timestamp is the index and its dtype is datetime64[ns]
# If you uncomment this, you'll see Temperature, Humidity, CO2, NH3, H2S, CH4

# Define time ranges for the specific test segments you want correlation matrices for
test_segments_info = [
    ("2025-04-17 12:27:01", "2025-04-17 12:42:05", 'Hay Test'),
    ("2025-04-17 12:42:05", "2025-04-17 13:02:02", 'Cowpatty Test'),
    ("2025-04-17 13:02:02", "2025-04-17 13:17:03", 'Dirt Test'),
    ("2025-04-17 13:17:03", "2025-04-17 13:33:25", 'Grass Test')
]

# --- Calculate and Plot Correlation Matrix for Each Test Segment ---

print("Generating Correlation Heatmaps for each test segment...")

for start_time_str, end_time_str, segment_label in test_segments_info:
    print(f"\nProcessing segment: {segment_label} ({start_time_str} to {end_time_str})")

    # Slice the DataFrame for the current segment using the index
    segment_df = df[start_time_str : end_time_str].copy() # Use .copy()

    # Check if the segment has data
    if segment_df.empty:
        print(f"Warning: No data found for {segment_label} in the DataFrame index range. Skipping correlation analysis.")
        continue # Skip to the next segment

    # Calculate the correlation matrix for this segment
    # min_periods=1 handles cases where there's very little data
    # .corr() is called on segment_df, which contains Temperature, Humidity, etc.
    correlation_matrix = segment_df.corr(min_periods=1)

    # Print the correlation matrix (optional, good for inspection)
    print("Correlation Matrix:")
    print(correlation_matrix)


    # Create a heatmap visualization for the correlation matrix
    # Adjusted figsize based on having 6 variables now (Temperature, Humidity, CO2, NH3, H2S, CH4)
    fig, ax = plt.subplots(figsize=(10, 8)) # Increased size for more columns/rows

    sns.heatmap(correlation_matrix,
                annot=True,     # Show the correlation values
                cmap='coolwarm',# Color map
                vmin=-1, vmax=1,# Consistent color limits
                fmt='.2f',      # Format annotations
                square=True,    # Make cells square
                linewidths=.5,  # Add lines
                ax=ax)          # Plot on the current axis

    # Set the title for the heatmap
    ax.set_title(f'Correlation Matrix: {segment_label}', fontsize=14)

    plt.tight_layout() # Adjust layout

    # Save the figure
    # Create a clean filename from the segment label
    filename_label = segment_label.replace(" ", "_").lower()
    plt.savefig(f'correlation_heatmap_{filename_label}.png')
    print(f"Saved heatmap: correlation_heatmap_{filename_label}.png")

    # Close the figure to free memory
    plt.close(fig)

print("\nFinished generating correlation heatmaps.")

# --- Removed/Commented Out Sections from Previous Task ---
# The following lines are from the previous task (plotting time series with segments/baselines)
# and are not needed for generating correlation heatmaps.

# baseline1_start_str = "2025-04-17 12:21:02" # Commented out as not used directly here
# baseline1_end_str = '2025-04-17 12:26:57'   # Commented out
# baseline2_start_str = "2025-04-17 13:34:04" # Commented out
# baseline2_end_str = "2025-04-17 13:53:47"   # Commented out

# --- Step 1 & 2: Isolate Baseline Data and Calculate Averages ---
# (These calculations are kept but not used in the heatmap plotting)
# baseline1_df = df[baseline1_start_str : baseline1_end_str].copy() # Commented out/can be kept if needed elsewhere
# ... baseline calculations ...
# baseline2_df = df[baseline2_start_str : baseline2_end_str].copy() # Commented out/can be kept if needed elsewhere
# ... baseline calculations ...

# --- Step 3: Define the data to be plotted ---
# This plot_df and subsequent plotting code is for time series, not heatmaps
# plot_start_str = baseline1_start_str # Commented out
# plot_end_str = baseline2_end_str     # Commented out
# plot_df = df[plot_start_str : plot_end_str].copy() # Commented out
# if plot_df.empty: ... # Commented out

# --- Define Segments for Coloring ---
# This 'segments' list with colors and labels was for time series plotting
# segments = [...] # Commented out or can be removed

# --- Plotting ---
# The entire plotting loop for time series subplots is replaced by the heatmap loop
# fig, axes = plt.subplots(...) # Commented out
# for i, col in enumerate(plot_df.columns): ... # Commented out
#    ax.plot(...) # Commented out
#    ax.axvspan(...) # Commented out
#    ax.axhline(...) # Commented out
#    ax.set_ylabel(col) # Commented out
#    ax.grid(...) # Commented out

# --- Configure the X-axis for date/time display ---
# This entire section is for time series plots and is not needed for heatmaps
# ax_bottom.xaxis.set_major_locator(...) # Commented out
# ax_bottom.xaxis.set_major_formatter(...) # Commented out
# ax_bottom.xaxis.set_minor_locator(...) # Commented out
# plt.xlabel('Timestamp') # Commented out
# fig.autofmt_xdate() # Commented out
# plt.tight_layout(...) # Commented out

# --- Add a combined legend outside the loop ---
# This legend is for the time series plot with segments/baselines
# handles, labels = [], [] # Commented out
# added_labels = set() # Commented out
# ... legend collection and creation ... # Commented out

# plt.savefig('cow_learn_segments_only.png') # Commented out
# plt.show() # Commented out

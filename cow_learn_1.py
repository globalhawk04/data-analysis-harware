import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.dates as mdates

# Load data and initial processing
df = pd.read_csv('sensor_readings_export.csv')
df = df.drop(columns=['ID'])

# Convert Timestamp to datetime *first* - this is essential!
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by timestamp (good practice)
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# Set Timestamp as the index - this creates the DatetimeIndex
df = df.set_index('Timestamp')

# Define time ranges
# Keep relevant time ranges defined
farm_start = "2025-04-17 12:15:04"
farm_base_line_1 = '2025-04-17 12:26:57' # Assuming this is the intended end of your baseline calculation
segmented_at_str = '2025-05-06 16:00:00' # String for the segmentation point
coffee_start = '2025-05-06 15:00:31' # Start of the plotting range
coffee_end = '2025-05-06 17:01:17' # End of the plotting range

# Convert segmentation timestamp to datetime object
segmented_at = pd.to_datetime(segmented_at_str)


# --- Step 1: Isolate the Baseline Data ---
# Create a DataFrame containing only the data for the baseline period
baseline_df = df[farm_start : farm_base_line_1]

# Check if baseline_df is empty
baseline_averages = None # Initialize baseline_averages
if baseline_df.empty:
    print(f"Warning: No data found in the baseline period ({farm_start} to {farm_base_line_1}). Cannot calculate baseline average.")
else:
    # --- Step 2: Calculate the Baseline Averages ---
    baseline_averages = baseline_df.mean()

    print(f"\nBaseline Averages ({farm_start} to {farm_base_line_1}):")
    print(baseline_averages)
    print("-" * 40)


# --- Step 3: Define the data to be plotted (Comparison Period - Coffee) ---
# Select the data range for the coffee phase
filtered_df = df[coffee_start : coffee_end].copy() # Use .copy() to avoid potential warnings later

if filtered_df.empty:
    print(f"Warning: No data found in the plotting period ({coffee_start} to {coffee_end}). Cannot plot.")
    exit()

# Also check if the segmentation point is within the plotting range
if segmented_at < filtered_df.index.min() or segmented_at > filtered_df.index.max():
    print(f"Warning: Segmentation point ({segmented_at_str}) is outside the plotting data range. Plotting without segmentation.")
    perform_segmentation = False
else:
     print(f"Plotting Data Range: {coffee_start} to {coffee_end}")
     print(f"Number of data points to plot: {len(filtered_df)}")
     print(f"Segmenting plot at: {segmented_at_str}")
     print("-" * 40)
     perform_segmentation = True


# --- Plotting ---
# Ensure we only plot columns that actually exist in filtered_df
cols_to_plot = filtered_df.columns
if len(cols_to_plot) == 0:
    print("No columns available to plot in filtered_df.")
    exit()

# Create the figure and subplots
fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=1, figsize=(12, len(cols_to_plot)*3), sharex=True)

fig.suptitle('Sensor Readings with Segmentation', fontsize=16) # Update title

# Ensure axes is iterable even if only one column
if len(cols_to_plot) == 1:
    axes = [axes] # Make it a list if only one subplot


for i, col in enumerate(cols_to_plot):
    ax = axes[i] # Get the current axis for this column

    if perform_segmentation:
        # --- Segment the data for the current column based on time ---
        data_before = filtered_df[filtered_df.index <= segmented_at]
        data_after = filtered_df[filtered_df.index > segmented_at]

        # --- Plot the segmented data ---
        # Plot the data BEFORE the segmentation point (e.g., blue)
        if not data_before.empty:
            ax.plot(data_before.index, data_before[col], color='blue', linestyle='-', label=f'{col} (Before)') # Add a distinct label

        # Plot the data AFTER the segmentation point (e.g., green)
        if not data_after.empty:
             # Ensure the line connects by potentially including the last point before
             # This helps visualize the transition properly
             connecting_point = filtered_df[filtered_df.index <= segmented_at].tail(1)
             if not connecting_point.empty:
                 combined_after = pd.concat([connecting_point, data_after])
             else:
                 combined_after = data_after

             ax.plot(combined_after.index, combined_after[col], color='green', linestyle='-', label=f'{col} (After)') # Add a distinct label

        # --- Add a vertical line marker at the segmentation point ---
        ax.axvline(x=segmented_at, color='gray', linestyle=':', linewidth=1, label='_nolegend_') # Use _nolegend_ to hide from legend

    else: # Plot as a single line if segmentation is not applicable or desired
         ax.plot(filtered_df.index, filtered_df[col], label=col)


    # --- Plot the Baseline Average Line ---
    # Check if baseline was calculated and average exists for this column
    if baseline_averages is not None and col in baseline_averages.index:
         # Plot a horizontal line at the baseline average for this column
         # Add label only for the first subplot to avoid redundant legend entries
         ax.axhline(y=baseline_averages[col], color='red', linestyle='--', label='Baseline Avg' if i == 0 else "_nolegend_")


    ax.set_ylabel(col)
    ax.legend(loc='upper left') # Legend will now show the 'Before' and 'After' segments + Baseline (if plotted)
    ax.grid(True)

# --- Configure the X-axis for date/time display ---
# Apply this to the bottom-most axis
ax_bottom = axes[-1]

# Set the major locator (e.g., every 10 minutes)
ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

# Set the major formatter to display HH:MM
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Optional: Set minor ticks for finer granularity (e.g., every 1 minute)
ax_bottom.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))


# Improve the layout and prevent labels overlapping
plt.xlabel('Timestamp') # Still set the overall xlabel on the bottom axis
fig.autofmt_xdate() # Auto-format the x-axis labels (rotate them)
plt.tight_layout(rect=[0,0.03,1,0.98]) # Adjust layout to make room for suptitle and labels

# Save the figure
plt.savefig('cow_learn_2_segmented.png') # Changed filename to reflect segmentation
#plt.show() # Display the plot
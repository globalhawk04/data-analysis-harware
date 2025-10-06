import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.dates as mdates # Import the dates module

# Load data and initial processing
df = pd.read_csv('sensor_learn_data.csv')
# Dropping multiple columns correctly
#df = df.drop(columns=['ID', 'Temperature', 'Humidity'])
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
#baseline1_start_str = "2025-04-17 12:21:02"
#baseline1_end_str = '2025-04-17 12:26:57'
hay_start_str =  "2025-04-17 12:27:01"
cowpatty_start_str = "2025-04-17 12:42:05"
dirt_start_str = "2025-04-17 13:02:02"
grass_start_str ="2025-04-17 13:17:03"
farm_end_str = "2025-04-17 13:33:25"
#baseline2_start_str = "2025-04-17 13:34:04"
#baseline2_end_str = "2025-04-17 13:53:47"


# --- Step 1 & 2: Isolate Baseline Data and Calculate Averages ---
# (These calculations are kept as they might be useful info, even if not plotted)

# Baseline 1 (Before Test)
baseline1_df = df[baseline1_start_str : baseline1_end_str] # Use string slicing
baseline1_averages = None # Initialize to None
if baseline1_df.empty:
    print(f"Warning: No data found in Baseline 1 period ({baseline1_start_str} to {baseline1_end_str}). Cannot calculate its average.")
else:
    baseline1_averages = baseline1_df.mean()
    print("\nBaseline 1 Averages (Before Test):")
    print(baseline1_averages)
    print("-" * 40)

# Baseline 2 (After Test)
baseline2_df = df[baseline2_start_str : baseline2_end_str] # Use string slicing
baseline2_averages = None # Initialize to None
if baseline2_df.empty:
    print(f"Warning: No data found in Baseline 2 period ({baseline2_start_str} to {baseline2_end_str}). Cannot calculate its average.")
else:
    baseline2_averages = baseline2_df.mean()
    print("\nBaseline 2 Averages (After Test):")
    print(baseline2_averages)
    print("-" * 40)


# --- Step 3: Define the data to be plotted ---
# Plot the full range including baselines and test periods
plot_start_str = baseline1_start_str
plot_end_str = baseline2_end_str

plot_df = df[plot_start_str : plot_end_str]

if plot_df.empty:
    print(f"Error: No data found in the overall plotting range ({plot_start_str} to {plot_end_str}). Exiting.")
    exit()

print(f"Plotting Data Range: {plot_start_str} to {plot_end_str}")
print(f"Number of data points to plot: {len(plot_df)}")
print("-" * 40)

# --- Define Segments for Coloring (using strings initially) ---
segments = [
    #(baseline1_start_str, baseline1_end_str, 'lightgray', 'Baseline 1 Period'), # Changed label slightly
    (hay_start_str, cowpatty_start_str, 'lightgreen', 'Hay Test Period'),
    (cowpatty_start_str, dirt_start_str, 'burlywood', 'Cowpatty Test Period'),
    (dirt_start_str, grass_start_str, 'saddlebrown', 'Dirt Test Period'),
    (grass_start_str, farm_end_str, 'olivedrab', 'Grass Test Period'),
    #(baseline2_start_str, baseline2_end_str, 'lightgray', 'Baseline 2 Period') # Same color as Baseline 1
]


# --- Plotting ---
# Create the figure and subplots
fig, axes = plt.subplots(nrows=len(plot_df.columns), ncols=1, figsize=(12,20), sharex=True)

fig.suptitle('Sensor Readings with Test Segment Highlighting', fontsize=16) # Update title

# Ensure axes is iterable even if only one column
if len(plot_df.columns) == 1:
    axes = [axes] # Make it a list if only one subplot


for i, col in enumerate(plot_df.columns):
    ax = axes[i] # Get the current axis for this column

    # --- Plot the time series data FIRST ---
    # This helps ensure the axis is properly initialized as a date axis
    ax.plot(plot_df.index, plot_df[col], label=col, zorder=3) # Give data line a higher zorder

    # --- Add Color/Shading for Each Segment ---
    # Now loop through segments and use converted datetime objects for axvspan
    for seg_start_str, seg_end_str, color, label in segments:
        try:
            # Convert segment strings to datetime objects for robustness with axvspan
            seg_start_dt = pd.to_datetime(seg_start_str)
            seg_end_dt = pd.to_datetime(seg_end_str)

            # Plot the vertical span for each defined segment
            ax.axvspan(seg_start_dt, seg_end_dt, color=color, alpha=0.3, zorder=0,
                       label=label if i == 0 else "") # Add label only to the first plot
        except ValueError as e:
             print(f"Could not convert segment dates {seg_start_str} to {seg_end_str} to datetime: {e}")
             print("Skipping this segment for plotting.")


    # --- Removed the ax.axhline calls for baseline averages ---
    # if baseline1_averages is not None and col in baseline1_averages.index:
    #      ax.axhline(y=baseline1_averages[col], color='red', linestyle='--', label='Baseline 1 Avg', zorder=4)
    # if baseline2_averages is not None and col in baseline2_averages.index:
    #      ax.axhline(y=baseline2_averages[col], color='blue', linestyle='-.', label='Baseline 2 Avg', zorder=4)


    ax.set_ylabel(col)
    # ax.legend(...) # Legend handled outside the loop
    ax.grid(True, zorder=1) # Ensure grid is above shading but below lines


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
plt.tight_layout(rect=[0,0.03,1,0.95]) # Adjust layout to make room for suptitle and combined legend

# --- Add a combined legend outside the loop ---
# This legend will now only include the data labels and segment labels
handles, labels = [], []
added_labels = set()
for ax in axes:
    current_handles, current_labels = ax.get_legend_handles_labels()

    for h, l in zip(current_handles, current_labels):
        if l not in added_labels:
            handles.append(h)
            labels.append(l)
            added_labels.add(l)

# Create the combined legend below the subplots
# Adjust ncol and bbox_to_anchor as needed
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0))


# Save the figure
plt.savefig('cow_learn_segments_only.png') # Changed filename again
#plt.show() # Display the plot
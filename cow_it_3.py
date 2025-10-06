import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For better date formatting
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats # For potential statistical tests later
import warnings
from matplotlib.patches import Patch # Needed for custom legend

# --- Configuration & Setup ---
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress common Seaborn FutureWarnings
sns.set_theme(style="whitegrid") # Set a clean default theme
plt.rcParams['figure.figsize'] = (15, 7) # Default figure size
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# --- Define Experimental Phases (Crucial Step!) ---
# NOTE: Assumes all timestamps are on the SAME DAY as the first data point.
# If data spans multiple days, you'll need to include the date.
# Using HH:MM format for easier definition. We'll combine with the date later.
# Make times timezone-aware if necessary (depends on source data)

# Convert provided times to datetime objects - *assuming same date as data start*
# We will determine the date after loading the data.
time_definitions = {
    "farm_start": "12:16",
    "hay_start": "12:27",
    "cowpatty_start": "12:42",
    "dirt_start": "13:02", # 1:02 PM
    "grass_start": "13:17", # 1:17 PM
    # We need an end time for the last segment - assume it continues for a reasonable duration
    # or ends when the data ends. Let's assume it ends when data ends for now.
}

# Sensor column names (consistent naming)
SENSOR_COLS = ['Temperature_F', 'Humidity_pct', 'CO2_ppm',
               'NH3_ppm', 'H2S_ppm', 'CH4_ppm']

# --- Load Data ---
try:
    df = pd.read_csv('sensor_readings_export.csv', parse_dates=['Timestamp'])
    print("Successfully loaded 'sensor_readings_export.csv'")
    print(f"Data ranges from {df['Timestamp'].min()} to {df['Timestamp'].max()}")
except FileNotFoundError:
    print("Error: 'sensor_readings_export.csv' not found.")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# --- Data Cleaning and Preparation ---

# 1. Timestamp already parsed. Check timezone awareness.
if df['Timestamp'].dt.tz is None:
    print("Timestamps are timezone naive. Assuming local time.")
    # Optionally localize: df['Timestamp'] = df['Timestamp'].dt.tz_localize('Your/Timezone')
else:
    print(f"Timestamps are timezone aware: {df['Timestamp'].dt.tz}")

# 2. Sort by Timestamp (ensure chronological order)
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# 3. Set Timestamp as the index (crucial for time series operations)
df = df.set_index('Timestamp')

# 4. Rename columns using the defined list
# Ensure we only rename the columns present (if fewer than len(SENSOR_COLS))
cols_to_rename = df.columns[:len(SENSOR_COLS)]
rename_dict = dict(zip(cols_to_rename, SENSOR_COLS))
df = df.rename(columns=rename_dict)
# Select only the renamed sensor columns + potentially others if needed later
# For now, let's assume we only care about the defined SENSOR_COLS
df = df[SENSOR_COLS]


# 5. Missing Value Check
print("\n--- Missing Value Check ---")
print(df.isnull().sum())
if df.isnull().values.any():
    print("Warning: Missing values detected. Consider imputation (e.g., ffill, interpolation).")
    # Example: df = df.ffill() # Forward fill

# 6. Infer Date and Create Full Timestamps for Segments
base_date = df.index.date[0] # Get the date from the first timestamp
print(f"Inferred base date for segments: {base_date}")

segment_times = {}
for name, time_str in time_definitions.items():
    try:
        # Combine the inferred date with the provided time strings
        full_datetime_str = f"{base_date} {time_str}"
        segment_times[name] = pd.to_datetime(full_datetime_str)
        # If your data is timezone-aware, make these timestamps aware too
        if df.index.tz is not None:
             # Localize the segment time to the data's timezone
             # Need to handle potential ambiguity/non-existence during DST transitions if applicable
             # Simple localization assuming no DST issues for this specific time:
             try:
                 segment_times[name] = segment_times[name].tz_localize(df.index.tzinfo)
             except Exception as tz_err: # Catch potential tz localization errors
                 print(f"Warning: Could not localize segment time {name} ({segment_times[name]}) to {df.index.tzinfo}. Error: {tz_err}. Proceeding timezone-naive.")
                 # Fallback or handle differently if timezone is critical
                 # Forcing UTC or a fixed offset might be an option if local timezone fails
                 try:
                      segment_times[name] = segment_times[name].tz_localize('UTC').tz_convert(df.index.tzinfo) # Example: Convert via UTC
                 except Exception as convert_err:
                      print(f"Further timezone conversion failed: {convert_err}")
                      segment_times[name] = pd.to_datetime(full_datetime_str) # Fallback to naive if conversion fails


    except ValueError:
        print(f"Error parsing time: {time_str} with date {base_date}. Please check format (HH:MM).")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred creating timestamp for {name}: {e}")
        exit()


# Sort segment times chronologically
segment_times = dict(sorted(segment_times.items(), key=lambda item: item[1]))
print("\nDefined Segment Start Times:")
for name, ts in segment_times.items():
    print(f"- {name}: {ts}")


# --- Add Segment Labels to the DataFrame ---
print("\n--- Adding Experimental Segment Labels ---")

# Define the time boundaries and corresponding labels (Keep your original label definition)
# We will redefine the bins and labels more dynamically based on segment_times
# and the data start/end.

# Let's redefine the bins based on explicit segment_times and the data start/end
# Bins should be [start1, start2), [start2, start3), ... [last_start, df.index.max()]
# We need to ensure a bin *starts* at df.index.min() for the first segment.

# Let's redefine labels to match these explicit intervals based on time_definitions
# Pre-Farm (before farm_start)
# Farm_Setup (farm_start to hay_start)
# Hay (hay_start to cowpatty_start)
# Cow_Patty (cowpatty_start to dirt_start)
# Dirt (dirt_start to grass_start)
# Grass_Forage (grass_start to end of data)

# New labels corresponding directly to intervals defined by time_definitions and data start/end
new_labels = []
bin_edges = [df.index.min()] # Start the first bin at the absolute minimum data time
prev_time_name = None

# Iterate through sorted segment start times to define intervals
sorted_segment_times_items = list(segment_times.items())
for i, (time_name, time_value) in enumerate(sorted_segment_times_items):
    # Add the current segment time as an end boundary for the previous interval
    # and a start boundary for the current interval.
    # We only add it if it's within the data range or slightly beyond the start.
    # Let's use only the defined times plus the data start/end as explicit bin edges.
    bin_edges.append(time_value)

# Ensure only unique and sorted bin edges
final_bins = sorted(list(set([df.index.min()] + list(segment_times.values()))))

# Add the end of the data range as the final bin edge
# Use a small offset to make the last interval [last_start, df.index.max()]
final_bins.append(df.index.max() + pd.Timedelta(microseconds=1))

# Ensure unique and sorted again after adding max
final_bins = sorted(list(set(final_bins)))


# Generate labels based on the final bins
# The number of labels should be len(final_bins) - 1
final_labels = []
# Get names of segments in the order of their start times
segment_names_ordered = [name.split('_')[0].capitalize() for name in segment_times.keys()]

for i in range(len(final_bins) - 1):
    start_time_bin = final_bins[i]
    end_time_bin = final_bins[i+1]

    label = f'Interval_{i+1}' # Default generic label

    # Find which defined segment starts *at* the start of this bin
    matching_segment_start = None
    for seg_name, seg_time in segment_times.items():
        # Check if the segment time is very close to the bin start time
        if abs(start_time_bin - seg_time) < pd.Timedelta(seconds=1):
             matching_segment_start = seg_name.split('_')[0].capitalize()
             break # Found a matching segment start

    if i == 0:
        # The first bin is always from df.index.min() up to the first defined time
        first_defined_name = sorted_segment_times_items[0][0].split('_')[0].capitalize()
        label = f'Pre-{first_defined_name}' # e.g., Pre-farm
    elif matching_segment_start:
         # If a segment starts exactly here, this interval is *that* segment's period
         # e.g., [farm_start, hay_start) is 'Farm_Setup' or 'Farm'
         # Let's use the name of the segment that *starts* at this bin boundary
         # And perhaps add '_Setup' for the first one after Pre-farm
         if i == 1 and final_labels[0].startswith('Pre-'): # This is the segment immediately following the first defined start
             label = f'{matching_segment_start}_Setup' # e.g., Farm_Setup
         else:
              label = matching_segment_start # e.g., Hay, Cowpatty, Dirt, Grass

    # Refine the last label to indicate "to End"
    if i == len(final_bins) - 2: # This is the last interval
        if matching_segment_start:
             label = f'{matching_segment_start}_to_End' # e.g., Grass_to_End
        elif final_labels and final_labels[-1].endswith('_Setup'):
             # If the previous label was 'X_Setup', this might be just 'X' or 'X_Continuation'
             # Or if the previous was 'Dirt', this is 'Grass_to_End' based on the *next* defined time
             # Let's just use the name of the segment that *started* the bin, appended with '_to_End'
             last_start_name_base = sorted_segment_times_items[-1][0].split('_')[0].capitalize()
             label = f'{last_start_name_base}_to_End'


    final_labels.append(label)


num_intervals = len(final_bins) - 1
print(f"Number of final bins: {len(final_bins)}")
# print(f"Final Bins: {final_bins}") # Uncomment for debugging
print(f"Number of intervals required: {num_intervals}")
print(f"Number of proposed new labels: {len(final_labels)}")


# Sanity check: Ensure number of labels matches number of intervals
if len(final_labels) != num_intervals:
    print(f"Error: Mismatch between number of intervals ({num_intervals}) and labels ({len(final_labels)}).")
    # Attempt to fix by padding labels if needed
    while len(final_labels) < num_intervals:
         final_labels.append(f'Auto_Interval_{len(final_labels)}')
    # Or truncate labels if needed (less likely to be correct)
    final_labels = final_labels[:num_intervals]
    print(f"Adjusted labels to match intervals: {len(final_labels)}")
    # print(f"Adjusted Labels: {final_labels}")


# --- Apply pd.cut ---
# Use include_lowest=True to ensure the very first data point is included
# Use right=False and ensure the last bin edge is slightly beyond df.index.max()
df['Segment'] = pd.cut(df.index,
                       bins=final_bins,
                       labels=final_labels, # Use the adjusted labels
                       right=False,          # Interval [start, end)
                       ordered=False,        # Treat as categorical initially
                       include_lowest=True) # Include the lowest value in the first bin


# Handle potential NaNs (though include_lowest=True should minimize this)
if df['Segment'].isnull().any():
    print("Warning: Some data points could not be assigned to a segment. Filling with 'Unknown'.")
    # Get existing categories before adding 'Unknown'
    existing_categories = list(df['Segment'].cat.categories)
    if 'Unknown' not in existing_categories:
         df['Segment'] = df['Segment'].cat.add_categories('Unknown')
         df['Segment'] = df['Segment'].fillna('Unknown')
else:
     # Add 'Unknown' category even if not used, for consistency in plotting functions if needed
     if 'Unknown' not in df['Segment'].cat.categories:
          df['Segment'] = df['Segment'].cat.add_categories('Unknown')


# --- Reorder segment levels ---
# The order is now naturally determined by the order of the final_labels list
# which was generated based on the sorted bin edges.
# We just need to make it an ordered categorical type.
# Ensure 'Unknown' is last if it exists
ordered_existing_segments = [label for label in final_labels if label in df['Segment'].unique().tolist()]
if 'Unknown' in df['Segment'].cat.categories and 'Unknown' in df['Segment'].unique().tolist() and 'Unknown' not in ordered_existing_segments:
     ordered_existing_segments.append('Unknown')
elif 'Unknown' in df['Segment'].cat.categories and 'Unknown' not in ordered_existing_segments:
     # Add 'Unknown' category even if it has no data points
     ordered_existing_segments.append('Unknown')


df['Segment'] = df['Segment'].astype(pd.CategoricalDtype(categories=ordered_existing_segments, ordered=True))


print("\nSegment Value Counts:")
print(df['Segment'].value_counts().sort_index()) # Sort by index for logical order


# --- Initial Data Overview (with Segment Context) ---
print("\n--- Cleaned Dataframe Head (with Segment) ---")
print(df.head())
print("\n--- Dataframe Info ---")
df.info() # Now includes the 'Segment' column

print("\n--- Descriptive Statistics (Overall) ---")
# Exclude non-numeric 'Segment' column from describe
print(df[SENSOR_COLS].describe().T)

# --- Segment-Based Descriptive Statistics ---
print("\n--- Descriptive Statistics by Segment ---")
# Use groupby to get stats for each sensor within each segment
# Check if 'Segment' column exists and is categorical before grouping
if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
    # Use observed=True if using Pandas >= 1.5.0 to avoid warnings/errors with empty categories
    try:
        segment_stats = df.groupby('Segment', observed=True)[SENSOR_COLS].agg(['mean', 'median', 'std', 'min', 'max'])
    except TypeError: # older pandas might not have observed kwarg
         segment_stats = df.groupby('Segment')[SENSOR_COLS].agg(['mean', 'median', 'std', 'min', 'max'])

    # Improve display if it gets too wide
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
         # Loop through each sensor for clearer printing
         for sensor in SENSOR_COLS:
             if sensor in segment_stats.columns: # Check if sensor exists in stats
                 print(f"\n--- Stats for: {sensor} ---")
                 # Select stats for the current sensor, drop levels for cleaner view
                 # print(segment_stats[sensor].sort_index().dropna(how='all')) # Print based on original category order if sorted
                 print(segment_stats[sensor].dropna(how='all')) # Print based on current category order
             else:
                 print(f"\n--- Skipping Stats for: {sensor} (Not found in segment_stats) ---")
else:
    print("Skipping segment statistics: 'Segment' column not found or not categorical.")


# --- Enhanced Time Series Plotting with Segment Visualization + Farm_Setup Avg Line ---
print("\n--- Plotting Sensor Data Over Time with Segments & Farm_Setup Avg ---")

# Find the 'Farm_Setup' segment name - now dynamically created, likely ends in '_Setup'
farm_setup_segment_name = None
for name in df['Segment'].cat.categories:
    if name.lower().endswith('_setup'):
         farm_setup_segment_name = name
         break

df_farm_setup = df[df['Segment'] == farm_setup_segment_name] if farm_setup_segment_name else pd.DataFrame()
farm_setup_avg = {}

if not df_farm_setup.empty:
    farm_setup_avg = df_farm_setup[SENSOR_COLS].mean().to_dict()
    print(f"\nCalculated Average for '{farm_setup_segment_name}':")
    for col, avg in farm_setup_avg.items():
        print(f"  {col}: {avg:.2f}")
else:
    print(f"\nWarning: No data found for segment '{farm_setup_segment_name}'. Skipping average line.")

n_sensors = len(SENSOR_COLS)
fig_ts, axes_ts = plt.subplots(nrows=n_sensors, ncols=1, figsize=(18, n_sensors * 4), sharex=True)
if n_sensors == 1: axes_ts = [axes_ts] # Ensure axes_ts is iterable for single sensor
fig_ts.suptitle('Sensor Readings Over Time by Experimental Segment with Farm Setup Avg', fontsize=18, y=1.01)

# Define colors for segments for consistency
if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
    unique_segments = df['Segment'].cat.categories.tolist() # Use ordered categories
    cmap = plt.get_cmap('tab20', max(len(unique_segments), 20)) # Use tab20 for more colors if needed
    segment_colors = {segment: cmap(i % cmap.N) for i, segment in enumerate(unique_segments)}

    lines_handles = [] # To collect handles for the line legend
    segment_patches = [] # To collect patches for the segment legend

    # Add segment patches to legend list *before* plotting lines for desired legend order
    present_segments = df['Segment'].unique().tolist()
    # Ensure present_segments follows the category order
    present_segments_ordered = [seg for seg in unique_segments if seg in present_segments]
    segment_patches = [Patch(color=segment_colors.get(seg, '#CCCCCC'), alpha=0.3, label=seg) for seg in present_segments_ordered]


    for i, col in enumerate(SENSOR_COLS):
        # Plot the main time series line
        line_handle, = axes_ts[i].plot(df.index, df[col], label=f'{col} (Raw)', linewidth=1, color='black', zorder=2)
        if i == 0: lines_handles.append(line_handle) # Add only one line handle for the legend

        # Add vertical lines for segment boundaries
        # Plot boundaries that fall within the data range (excluding min/max which are edges)
        segment_boundaries_for_plot = [t for t in final_bins if t > df.index.min() and t < df.index.max()]
        for t in segment_boundaries_for_plot:
            axes_ts[i].axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=3)

        # Add shaded background colors for segments (more visual separation)
        # Iterate through the *ordered* unique segments found in the data
        for k, segment_name in enumerate(unique_segments):
            segment_data = df[df['Segment'] == segment_name]
            if not segment_data.empty:
                start = segment_data.index.min()
                end = segment_data.index.max()
                # Ensure end is > start for plotting span
                if start < end:
                    axes_ts[i].axvspan(start, end, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=1) # No label here, using patches


        # Add the Farm_Setup Average line
        if farm_setup_segment_name and farm_setup_segment_name in df['Segment'].cat.categories and farm_setup_avg and col in farm_setup_avg and pd.notna(farm_setup_avg[col]):
             # Find the start time of the Farm_Setup segment
             farm_setup_start_time = df[df['Segment'] == farm_setup_segment_name].index.min()
             farm_setup_end_time = df[df['Segment'] == farm_setup_segment_name].index.max()
             if pd.isna(farm_setup_start_time) or pd.isna(farm_setup_end_time) or farm_setup_start_time >= farm_setup_end_time:
                 print(f"Warning: Farm Setup segment '{farm_setup_segment_name}' start/end times not suitable for plotting avg line range.")
                 # Plot the line across the whole plot if specific segment range is problematic
                 avg_line_handle, = axes_ts[i].plot([df.index.min(), df.index.max()],
                                                   [farm_setup_avg[col], farm_setup_avg[col]],
                                                   color='green', linestyle='-', linewidth=2, alpha=0.7, zorder=4,
                                                   label=f'{farm_setup_segment_name} Avg' if i == 0 else "") # Label only on first plot

             else:
                 # Plot the line *only* across the Farm_Setup segment duration
                 # Use the actual min/max of the segment data index
                 avg_line_handle, = axes_ts[i].plot([farm_setup_start_time, farm_setup_end_time],
                                                   [farm_setup_avg[col], farm_setup_avg[col]],
                                                   color='green', linestyle='-', linewidth=2, alpha=0.7, zorder=4,
                                                   label=f'{farm_setup_segment_name} Avg' if i == 0 else "") # Label only on first plot


             if i == 0: lines_handles.append(avg_line_handle) # Add avg line handle to legend

        axes_ts[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
        axes_ts[i].grid(True, which='major', linestyle=':', linewidth=0.5)
        # Legend will be consolidated at the top

    # Consolidate legends - put one legend at the top
    # Filter out handles with empty labels if needed
    valid_handles = [h for h in lines_handles + segment_patches if h.get_label() and not h.get_label().startswith('_')] # Filter out default matplotlib labels
    fig_ts.legend(handles=valid_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(valid_handles), 8)) # Adjust ncol

    plt.xlabel('Timestamp')
    # Improve date formatting on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=20)) # Adjust number of ticks
    plt.gcf().autofmt_xdate() # Rotate date labels

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout further if needed
    plt.savefig('cow_timeseries_segmented_farm_setup_avg.png', dpi=300)
    print("Saved segmented time series plots with Farm Setup Avg to 'cow_timeseries_segmented_farm_setup_avg.png'")
    # plt.show()

else:
    print("Skipping time series plot: 'Segment' column not properly set up.")


# --- New Chart Comparing Pre-Lab Average to Entire Time Series ---
print("\n--- Plotting Sensor Data Over Time with Pre-Lab Avg ---")

# Calculate the average for the 'Pre-Lab' segment
prelab_segment_name = df['Segment'].cat.categories[0] if len(df['Segment'].cat.categories) > 0 else None # Assuming first segment is Pre-Lab
df_prelab = df[df['Segment'] == prelab_segment_name] if prelab_segment_name else pd.DataFrame()
prelab_avg = {}

if not df_prelab.empty:
    prelab_avg = df_prelab[SENSOR_COLS].mean().to_dict()
    print(f"\nCalculated Average for '{prelab_segment_name}':")
    for col, avg in prelab_avg.items():
        print(f"  {col}: {avg:.2f}")
    prelab_start_time = df_prelab.index.min()
    prelab_end_time = df_prelab.index.max()
else:
    print(f"\nWarning: No data found for segment '{prelab_segment_name}'. Skipping Pre-Lab average line plots.")


if prelab_segment_name and not df_prelab.empty and prelab_avg:
    fig_prelab, axes_prelab = plt.subplots(nrows=n_sensors, ncols=1, figsize=(18, n_sensors * 4), sharex=True)
    if n_sensors == 1: axes_prelab = [axes_prelab] # Ensure axes_prelab is iterable
    fig_prelab.suptitle('Sensor Readings Over Time Compared to Pre-Lab Average', fontsize=18, y=1.01)

    lines_handles_prelab = [] # To collect handles for the line legend
    segment_patches_prelab = [] # To collect patches for the segment legend

    # Add segment patches to legend list (re-using segment_colors from previous plot)
    present_segments = df['Segment'].unique().tolist()
    # Ensure present_segments follows the category order
    present_segments_ordered = [seg for seg in unique_segments if seg in present_segments] # Use unique_segments from previous plot for order
    segment_patches_prelab = [Patch(color=segment_colors.get(seg, '#CCCCCC'), alpha=0.3, label=seg) for seg in present_segments_ordered]


    for i, col in enumerate(SENSOR_COLS):
        # Plot the main time series line
        line_handle, = axes_prelab[i].plot(df.index, df[col], label=f'{col} (Raw)', linewidth=1, color='black', zorder=2)
        if i == 0: lines_handles_prelab.append(line_handle) # Add only one line handle

        # Add segment background colors/lines for context (re-using logic)
        segment_boundaries_for_plot = [t for t in final_bins if t > df.index.min() and t < df.index.max()]
        for t in segment_boundaries_for_plot:
            axes_prelab[i].axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=3)

        for k, segment_name in enumerate(unique_segments):
            segment_data = df[df['Segment'] == segment_name]
            if not segment_data.empty:
                start = segment_data.index.min()
                end = segment_data.index.max()
                if start < end:
                    axes_prelab[i].axvspan(start, end, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=1)


        # Add the Pre-Lab Average line
        if col in prelab_avg and pd.notna(prelab_avg[col]):
            # Plot the line across the entire data range
            avg_line_handle, = axes_prelab[i].plot([df.index.min(), df.index.max()],
                                                   [prelab_avg[col], prelab_avg[col]],
                                                   color='purple', linestyle='--', linewidth=2, alpha=0.7, zorder=4,
                                                   label=f'{prelab_segment_name} Avg' if i == 0 else "") # Label only on first plot
            if i == 0: lines_handles_prelab.append(avg_line_handle) # Add avg line handle

        axes_prelab[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
        axes_prelab[i].grid(True, which='major', linestyle=':', linewidth=0.5)
        # Legend will be consolidated at the top

    # Consolidate legends - put one legend at the top
    valid_handles_prelab = [h for h in lines_handles_prelab + segment_patches_prelab if h.get_label() and not h.get_label().startswith('_')]
    fig_prelab.legend(handles=valid_handles_prelab, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(valid_handles_prelab), 8)) # Adjust ncol

    plt.xlabel('Timestamp')
    # Improve date formatting on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=20)) # Adjust number of ticks
    plt.gcf().autofmt_xdate() # Rotate date labels

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout further if needed
    plt.savefig('cow_timeseries_segmented_prelab_avg.png', dpi=300)
    print("Saved time series plots with Pre-Lab Avg comparison to 'cow_timeseries_segmented_prelab_avg.png'")
    # plt.show()
else:
    print("Skipping Pre-Lab average comparison plots: Pre-Lab segment not found or no data.")


# --- Comparative Distribution Analysis (Box Plots/Violin Plots per Segment) ---
print("\n--- Comparing Sensor Distributions Across Segments ---")

if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype) and not df['Segment'].isnull().all():
    n_cols_dist = 2
    n_rows_dist = (n_sensors + n_cols_dist - 1) // n_cols_dist
    fig_dist, axes_dist = plt.subplots(n_rows_dist, n_cols_dist, figsize=(16, n_rows_dist * 5), sharex=False) # Don't share x for boxplots usually
    axes_dist = axes_dist.flatten()

    # Get the ordered categories for plotting
    plot_order = df['Segment'].cat.categories.tolist()
    # Use the same colors as the time series plot
    palette = {seg: segment_colors.get(seg, '#CCCCCC') for seg in plot_order}

    for i, col in enumerate(SENSOR_COLS):
        # Check if column exists in df and has data
        if col in df.columns and not df[col].isnull().all():
            # Use seaborn for easy grouped boxplots or violinplots
            sns.boxplot(x='Segment', y=col, data=df, ax=axes_dist[i],
                        palette=palette, order=plot_order, showfliers=False) # Hide outliers initially
            # Or use violin plot:
            # sns.violinplot(x='Segment', y=col, data=df, ax=axes_dist[i], palette=palette, order=plot_order, inner='quartile')

            axes_dist[i].set_title(f'{col} Distribution by Segment')
            axes_dist[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
            axes_dist[i].set_xlabel('Experimental Segment')
            # REMOVE 'ha' from tick_params - ha is for text objects, not the tick parameters themselves
            # axes_dist[i].tick_params(axis='x', rotation=45, ha='right') # <-- THIS LINE CAUSED THE ERROR
            axes_dist[i].tick_params(axis='x', rotation=45) # Corrected line - remove ha

            # Correctly set rotation and horizontal alignment for the tick labels (Text objects)
            plt.setp(axes_dist[i].get_xticklabels(), rotation=45, ha="right") # Keep this line, it's correct


        else:
            print(f"Skipping distribution plot for {col}: Column not found or no valid data.")
            # Optionally hide the axis if the column is missing
            axes_dist[i].set_visible(False)


    # Hide unused subplots
    for j in range(i + 1, len(axes_dist)):
        fig_dist.delaxes(axes_dist[j])

    plt.tight_layout()
    plt.savefig('cow_segment_boxplots.png', dpi=300)
    print("Saved segment comparison box plots to 'cow_segment_boxplots.png'")
    # plt.show()
else:
    print("Skipping distribution plots: 'Segment' column not properly set up.")


# --- Correlation Analysis (Overall and within key segments) ---
print("\n--- Correlation Analysis ---")

# 1. Overall Correlation
print("\n--- Overall Correlation Matrix ---")
# Ensure only numeric columns are used
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    print("No numeric columns found for correlation analysis.")
else:
    correlation_matrix_overall = df[numeric_cols].corr()
    print(correlation_matrix_overall)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_overall, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Overall Correlation Matrix of Sensor Readings', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('cow_correlation_heatmap_overall.png')
    print("Saved overall correlation heatmap to 'cow_correlation_heatmap_overall.png'")
    # plt.show()


    # 2. Correlation within specific segments (e.g., Lab vs. Farm)
    if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
        # Identify key segments for comparison
        segments_to_compare_corr = []
        # Add segments based on actual created categories
        if df['Segment'].cat.categories.size > 0:
            segments_to_compare_corr.append(df['Segment'].cat.categories[0]) # First segment (usually pre-farm)
        # Try to find the first segment ending in '_Setup'
        setup_seg = next((s for s in df['Segment'].cat.categories if s.lower().endswith('_setup')), None)
        if setup_seg and setup_seg not in segments_to_compare_corr:
             segments_to_compare_corr.append(setup_seg)
        # Add other specific names if they exist
        specific_names = ['Hay', 'Cowpatty', 'Dirt', 'Grass_to_End', 'Unknown']
        for name in specific_names:
            if name in df['Segment'].cat.categories and name not in segments_to_compare_corr:
                 segments_to_compare_corr.append(name)

        print(f"\nComparing correlations for segments: {segments_to_compare_corr}")

        for segment_name in segments_to_compare_corr:
            if segment_name in df['Segment'].cat.categories:
                df_segment = df[df['Segment'] == segment_name]
                if not df_segment.empty and len(df_segment.index) > 1:
                     print(f"\n--- Correlation Matrix ({segment_name} Segment) ---")
                     # Ensure only numeric columns with variance > 0 within the segment are used
                     segment_numeric_cols = df_segment.select_dtypes(include=np.number)
                     # Filter out columns that are constant or NaN within this segment
                     segment_numeric_cols = segment_numeric_cols.loc[:, segment_numeric_cols.std() > 0.001] # Check std > small value
                     segment_numeric_cols = segment_numeric_cols.dropna(axis=1, how='all') # Drop columns with all NaNs

                     if not segment_numeric_cols.empty and len(segment_numeric_cols.columns) > 1:
                        correlation_matrix_segment = segment_numeric_cols.corr()
                        print(correlation_matrix_segment)
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(correlation_matrix_segment, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1) # Keep scale consistent
                        plt.title(f'Correlation Matrix ({segment_name} Segment)', fontsize=16)
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        safe_segment_name = segment_name.replace(' ', '_').replace('-', '_').replace('.', '') # Sanitize for filename
                        plt.savefig(f'cow_correlation_heatmap_{safe_segment_name}.png')
                        print(f"Saved {segment_name} correlation heatmap to cow_correlation_heatmap_{safe_segment_name}.png")
                        # plt.show()
                     else:
                         print(f"Skipping {segment_name} correlation: Not enough numeric columns with variance.")
                else:
                     print(f"\nSkipping {segment_name} correlation: Not enough data.")
            # else: print handled by the loop constructing segments_to_compare_corr


    else:
        print("Skipping segmented correlation: 'Segment' column not available.")


# --- Resampling and Rolling Stats (Can still be useful) ---
# Apply these *after* segmentation if you want to analyze smoothed trends within segments,
# or apply them to the whole dataset as before to see overall smoothing.
# Applying to the whole dataset here:

# Resample to 1-minute averages
resample_freq = '1T'
# Recalculate on original numeric data
if not numeric_cols:
    print("Skipping resampling: No numeric columns.")
else:
    # Resample only numeric columns
    df_resampled = df[numeric_cols].resample(resample_freq).mean()

    print(f"\n--- Data Resampled to {resample_freq} Averages ---")
    print(df_resampled.head())

    # Plotting Resampled Data (Example for one sensor)
    sensor_to_plot_resample = 'CO2_ppm'
    if sensor_to_plot_resample in df.columns and sensor_to_plot_resample in df_resampled.columns:
        plt.figure(figsize=(18, 5))
        plt.plot(df_resampled.index, df_resampled[sensor_to_plot_resample], label=f'{sensor_to_plot_resample} ({resample_freq} Avg)', color='red', linewidth=1.5)
        plt.plot(df.index, df[sensor_to_plot_resample], label=f'{sensor_to_plot_resample} (Raw)', alpha=0.3, color='blue', linewidth=0.5)
        plt.title(f'{sensor_to_plot_resample} Comparison: Raw vs. {resample_freq} Average')
        plt.ylabel(sensor_to_plot_resample.replace('_', ' '))
        # Add segment lines/shading to resampled plot too for context (using segment_colors)
        if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
             # Use unique_segments from previous plot for order
             unique_segments_for_plot = [seg for seg in df['Segment'].cat.categories if seg in segment_colors]
             for k, segment_name in enumerate(unique_segments_for_plot):
                 segment_data = df[df['Segment'] == segment_name]
                 if not segment_data.empty:
                     start = segment_data.index.min()
                     end = segment_data.index.max()
                     if start < end:
                          plt.axvspan(start, end, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=1) # Add shading
             segment_boundaries_for_plot = [t for t in final_bins if t > df.index.min() and t < df.index.max()]
             for t in segment_boundaries_for_plot:
                  plt.axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=3) # Add lines


        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        safe_sensor_name = sensor_to_plot_resample.replace('_', '').replace(' ', '')
        plt.savefig(f'cow_{safe_sensor_name}_resample_comparison_{resample_freq}.png')
        print(f"Saved {sensor_to_plot_resample} resample comparison plot.")
        # plt.show()
    else:
        print(f"Skipping resample plot for {sensor_to_plot_resample}: Column not found in original or resampled data.")


    # Rolling Statistics (Example for one sensor)
    # Estimate sample rate (can be inaccurate if gaps exist)
    time_diffs = df.index.to_series().diff().dropna() # Drop first NaN
    sampling_rate_seconds = time_diffs.median().total_seconds() if not time_diffs.empty else 5 # Default to 5s if no diffs

    if pd.notna(sampling_rate_seconds) and sampling_rate_seconds > 0:
        print(f"Estimated sampling rate: {sampling_rate_seconds:.2f} seconds")
        window_size_minutes = 5 # 5 minutes
        # Calculate window size in samples (handle potential zero sampling rate)
        window_size_samples = max(1, int((window_size_minutes * 60) / sampling_rate_seconds)) if sampling_rate_seconds > 0 else 10 # Default if rate unknown
        window_label = f'{window_size_minutes}min'

        print(f"\n--- Calculating Rolling Statistics (Window: {window_label} / {window_size_samples} samples) ---")
        sensor_to_roll = 'NH3_ppm'

        if sensor_to_roll in df.columns and not df[sensor_to_roll].isnull().all():
            # Only calculate rolling stats if there's valid data
            rolling_mean = df[sensor_to_roll].rolling(window=window_size_samples, min_periods=1).mean()
            rolling_std = df[sensor_to_roll].rolling(window=window_size_samples, min_periods=1).std()

            plt.figure(figsize=(18, 6))
            plt.plot(df.index, df[sensor_to_roll], color='lightblue', alpha=0.6, label=f'{sensor_to_roll} (Raw)', linewidth=0.8)
            plt.plot(rolling_mean.index, rolling_mean, color='blue', label=f'Rolling Mean ({window_label})', linewidth=1.5)
            plt.ylabel(sensor_to_roll.replace('_', ' '))
            plt.title(f'{sensor_to_roll}: Raw Data and Rolling Statistics ({window_label})')

            # Add Std Dev on secondary axis
            ax2 = plt.gca().twinx()
            ax2.plot(rolling_std.index, rolling_std, color='red', linestyle='--', label=f'Rolling Std Dev ({window_label})', linewidth=1)
            ax2.set_ylabel('Rolling Std Dev', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.spines['right'].set_color('red')

            # Combine legends
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.gca().legend(lines + lines2, labels + labels2, loc='upper left')

            # Add segment lines/shading
            if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
                 # Use unique_segments from previous plot for order
                 unique_segments_for_plot = [seg for seg in df['Segment'].cat.categories if seg in segment_colors]
                 for k, segment_name in enumerate(unique_segments_for_plot):
                     segment_data = df[df['Segment'] == segment_name]
                     if not segment_data.empty:
                         start = segment_data.index.min()
                         end = segment_data.index.max()
                         if start < end:
                              plt.axvspan(start, end, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=0) # Use zorder 0 so lines are on top
                 segment_boundaries_for_plot = [t for t in final_bins if t > df.index.min() and t < df.index.max()]
                 for t in segment_boundaries_for_plot:
                      plt.axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=0) # Use zorder 0

            plt.grid(True, axis='y', linestyle=':')
            plt.xlabel('Timestamp')
            plt.tight_layout()
            safe_sensor_name = sensor_to_roll.replace('_', '').replace(' ', '')
            plt.savefig(f'cow_{safe_sensor_name}_rolling_stats_{window_label}.png')
            print(f"Saved {sensor_to_roll} rolling stats plot.")
            # plt.show()
        else:
            print(f"Skipping rolling stats plot for {sensor_to_roll}: Column not found or no valid data.")
    else:
        print("Skipping rolling stats: Could not determine sampling rate or rate is zero.")



# --- Time Series Decomposition (Consider decomposing key segments) ---
# Decomposing the *entire* series might mix different behaviors.
# Consider decomposing a long, relatively stable segment like 'Lab' or a 'Farm' period if long enough.

if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
    variable_to_decompose = 'Temperature_F'
    # Find a suitable segment for decomposition - maybe the one with most data points after pre-farm
    # Get all categories except the first one (Pre-farm) and 'Unknown'
    categories_to_consider = [s for s in df['Segment'].cat.categories if s != df['Segment'].cat.categories[0] and s != 'Unknown']
    suitable_segments = [s for s in categories_to_consider if not df[df['Segment']==s].empty]

    if suitable_segments:
         # Pick the segment with the most data points from the suitable ones
         segment_counts = df['Segment'].value_counts()
         # Ensure we only pick from suitable_segments that are actually in the counts
         suitable_segments_with_data = [s for s in suitable_segments if s in segment_counts.index]
         if suitable_segments_with_data:
             segment_to_decompose = segment_counts[suitable_segments_with_data].idxmax()
             print(f"\nAutomatically selected segment '{segment_to_decompose}' for decomposition based on size (excluding first segment and Unknown).")
         else:
              segment_to_decompose = None
              print("\nNo suitable segment found with data for decomposition (excluding first segment and Unknown).")
    else:
         segment_to_decompose = None
         print("\nNo suitable segment found for decomposition (excluding first segment and Unknown).")


    if segment_to_decompose and variable_to_decompose in df.columns:
        df_segment_decomp = df[df['Segment'] == segment_to_decompose][variable_to_decompose].dropna()

        # Choose period carefully. Based on estimated sampling rate
        # Use the calculated sampling_rate_seconds from Rolling Stats section if available
        decomposition_period = None
        if 'sampling_rate_seconds' in locals() and pd.notna(sampling_rate_seconds) and sampling_rate_seconds > 0:
             samples_per_hour = int(3600 / sampling_rate_seconds)
             samples_per_10_min = int(600 / sampling_rate_seconds)
             # Choose a reasonable period, e.g., 10 minutes or 1 hour, if segment is long enough
             # Ensure period is at least 2 and less than half the data points in the segment
             # Prioritize 10min if possible, otherwise try 1hr/6 (arbitrary fallback)
             potential_periods = [samples_per_10_min, samples_per_hour // 6]
             for p in potential_periods:
                  if p >= 2 and (2 * p + 1) <= len(df_segment_decomp):
                       decomposition_period = p
                       break
             if decomposition_period is None: # If neither potential period fits length
                  decomposition_period = max(2, len(df_segment_decomp) // 4) # Fallback: Use 1/4 segment length, min 2
                  # Ensure it's still valid for decomposition calculation
                  if (2 * decomposition_period + 1) > len(df_segment_decomp):
                       decomposition_period = max(2, (len(df_segment_decomp) - 1) // 2) # Ensure length >= 2*period + 1
                       if decomposition_period < 2: decomposition_period = 2


        else:
             decomposition_period = min(max(2, 120), (len(df_segment_decomp) - 1) // 2) # Fallback: 10-minute period assuming ~12 samples/min (5s rate), ensure valid range

        min_len_for_decomp = 2 * decomposition_period + 1 # Decomposition requires at least 2 periods + 1 point

        print(f"\n--- Attempting Decomposition for '{variable_to_decompose}' in Segment '{segment_to_decompose}' ---")
        print(f"Segment length: {len(df_segment_decomp)} data points.")
        print(f"Selected period: {decomposition_period} samples.")
        print(f"Required minimum length for period {decomposition_period}: {min_len_for_decomp} data points.")


        if len(df_segment_decomp) >= min_len_for_decomp and decomposition_period >= 2:
            try:
                # Ensure the index has a frequency if possible, or handle decomposition without it
                # df_segment_decomp = df_segment_decomp.asfreq(pd.Timedelta(seconds=sampling_rate_seconds)) # Might introduce NaNs
                # For decomposition to work reliably, the time series should be regular.
                # seasonal_decompose can sometimes handle irregular series, but setting freq helps.
                # However, setting freq on an irregular series introduces NaNs which break decomposition.
                # It's often better to decompose an irregular series directly if statsmodels supports it well enough for the data.
                # Let's try without setting freq first.
                decomposition = seasonal_decompose(df_segment_decomp, model='additive', period=decomposition_period, extrapolate_trend='freq') # extrapolate_trend helps at ends
                fig_decomp = decomposition.plot()
                fig_decomp.set_size_inches(14, 10)
                fig_decomp.suptitle(f'Time Series Decomposition of {variable_to_decompose}\n(Segment: {segment_to_decompose}, Period: {decomposition_period})', y=1.05)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                safe_var_name = variable_to_decompose.replace('_', '')
                safe_seg_name = segment_to_decompose.replace(' ', '_').replace('-', '_').replace('.', '')
                plt.savefig(f'cow_decomposition_{safe_var_name}_{safe_seg_name}.png')
                print(f"Saved decomposition plot to 'cow_decomposition_{safe_var_name}_{safe_seg_name}.png'")
                # plt.show()
            except ValueError as e:
                print(f"Could not perform decomposition for segment '{segment_to_decompose}', Error: {e}")
                print("This might be due to insufficient length, NaNs, or irregular time intervals within the segment.")
                print("Try resampling the segment data to a regular frequency before decomposition.")
            except Exception as e:
                 print(f"An unexpected error occurred during decomposition: {e}")
        else:
            print(f"Skipping Decomposition for segment '{segment_to_decompose}': Insufficient data length ({len(df_segment_decomp)}) for the chosen period ({decomposition_period}). Required: {min_len_for_decomp}.")
    elif segment_to_decompose:
         print(f"Skipping decomposition: Variable '{variable_to_decompose}' not found in dataframe.")
    # else: print handled above


else:
    print("Skipping decomposition: 'Segment' column not available or not categorical.")


# --- Potential Next Steps & Advanced Analysis ---
print("\n--- Analysis Complete ---")
print("Consider further analysis:")
print("1.  Statistical Tests: Use ANOVA or t-tests (or non-parametric equivalents like Kruskal-Wallis/Mann-Whitney U) to formally compare distributions between key segments (e.g., Lab vs. Hay vs. Grass for NH3_ppm).")
print("2.  Outlier Investigation: Analyze the outliers potentially hidden in box plots (re-run with showfliers=True). Are they errors or significant events?")
print("3.  Change Point Detection: Use algorithms (e.g., from `ruptures` library) to automatically detect shifts in sensor behavior and compare with defined segments.")
print("4.  Feature Engineering: Create new features (e.g., rates of change `df[col].diff()`, ratios between gases `df['NH3_ppm'] / df['CO2_ppm']`).")
print("5.  Environmental Correlation: If available, correlate sensor data with external factors (e.g., ambient temperature, ventilation changes, animal activity logs).")
print("6.  Interactive Visualizations: Use libraries like Plotly or Bokeh for plots you can zoom, pan, and hover over for details.")
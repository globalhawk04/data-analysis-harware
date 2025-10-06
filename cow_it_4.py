import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # For better date formatting
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats  # For potential statistical tests later
import warnings
from matplotlib.patches import Patch  # Needed for custom legend

# --- Configuration & Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress common Seaborn FutureWarnings
sns.set_theme(style="whitegrid")  # Set a clean default theme
plt.rcParams['figure.figsize'] = (15, 7)  # Default figure size
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
    "farm_start": "12:16", # This starts the Farm_Setup segment
    "hay_start": "12:27",
    "cowpatty_start": "12:42",
    "dirt_start": "13:02",  # 1:02 PM
    "grass_start": "13:17",  # 1:17 PM # This starts the Grass_to_End segment
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
base_date = df.index.date[0]  # Get the date from the first timestamp
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
            try:
                segment_times[name] = segment_times[name].tz_localize(df.index.tzinfo)
            except Exception as tz_err:  # Catch potential tz localization errors
                print(f"Warning: Could not localize segment time {name} ({segment_times[name]}) to {df.index.tzinfo}. Error: {tz_err}. Proceeding timezone-naive.")
                try:
                    segment_times[name] = segment_times[name].tz_localize('UTC').tz_convert(df.index.tzinfo)  # Example: Convert via UTC
                except Exception as convert_err:
                    print(f"Further timezone conversion failed: {convert_err}")
                    segment_times[name] = pd.to_datetime(full_datetime_str)  # Fallback to naive if conversion fails


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

# Use the sorted time_definitions keys to generate base labels
# Map time definition keys to desired segment labels
segment_label_mapping = {
    "farm_start": "Farm_Setup",
    "hay_start": "Hay",
    "cowpatty_start": "Cowpatty",
    "dirt_start": "Dirt",
    "grass_start": "Grass_to_End" # Use the label generated by the previous logic
}


# Create bins from df.index.min() and sorted segment times
# Ensure only times within the data range are used as bin edges
valid_segment_times_in_data_range = [t for t in segment_times.values() if t >= df.index.min() and t <= df.index.max()]

final_bins = sorted(list(set([df.index.min()] + valid_segment_times_in_data_range)))

# Add the end of the data range as the final bin edge
final_bins.append(df.index.max() + pd.Timedelta(microseconds=1)) # Add a tiny offset

# Ensure unique and sorted again
final_bins = sorted(list(set(final_bins)))


# Create labels based on which time_definition starts each interval
final_labels = []
for i in range(len(final_bins) - 1):
    start_of_interval = final_bins[i]
    end_of_interval = final_bins[i+1] # Not strictly needed for label, but good for context

    label = f'Auto_Interval_{i+1}' # Default fallback

    if i == 0:
        # The first segment is always from data start to the first defined time
        first_defined_key = list(segment_times.keys())[0] # Get the key of the chronologically first time
        first_defined_base_name = segment_label_mapping.get(first_defined_key, first_defined_key.split('_')[0].capitalize())
        # Remove _Setup if it was added, we want the "Pre-" to be before the base name
        first_defined_base_name = first_defined_base_name.replace('_Setup', '')
        label = f'Pre-{first_defined_base_name}'
    else:
        # Find the time_definition key that corresponds to the start of this interval
        matching_key = None
        for key, time_val in segment_times.items():
            if abs(start_of_interval - time_val) < pd.Timedelta(seconds=1):
                matching_key = key
                break

        if matching_key:
             # Use the predefined mapping, or fallback to splitting the key name
             label = segment_label_mapping.get(matching_key, matching_key.split('_')[0].capitalize())
             # Special handling for the last segment name if it's 'Grass_to_End' or similar, ensure it's assigned correctly
             if matching_key == list(segment_times.keys())[-1] and final_bins[-2] == start_of_interval:
                  label = segment_label_mapping.get(matching_key, matching_key.split('_')[0].capitalize() + '_to_End') # Use the mapping or add _to_End as fallback

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
         final_labels.append(f'Auto_Interval_{len(final_labels)}') # Add generic labels for missing ones
    final_labels = final_labels[:num_intervals] # Truncate if somehow too many (less likely)
    print(f"Adjusted labels to match intervals: {len(final_labels)}")
    # print(f"Adjusted Labels: {final_labels}")


# --- Apply pd.cut ---
# Use include_lowest=True to ensure the very first data point is included
# Use right=False and ensure the last bin edge is slightly beyond df.index.max()
df['Segment'] = pd.cut(df.index,
                       bins=final_bins,
                       labels=final_labels,  # Use the adjusted labels
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
# This plot visualizes ALL generated segments.
print("\n--- Plotting Sensor Data Over Time by Experimental Segment with Farm Setup Avg ---")

# Find the 'Farm_Setup' segment name
farm_setup_segment_name = 'Farm_Setup' # Explicitly look for this label based on mapping

df_farm_setup = df[df['Segment'] == farm_setup_segment_name] if farm_setup_segment_name in df['Segment'].cat.categories else pd.DataFrame()
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

# Define colors for segments for consistency across all plots
segment_colors = {}
if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
    unique_segments_with_data = df['Segment'].dropna().unique().tolist() # Get unique segments with actual data
    # Ensure unique_segments list follows the defined category order for consistent coloring
    unique_segments_ordered = [cat for cat in df['Segment'].cat.categories if cat in unique_segments_with_data]

    cmap = plt.get_cmap('tab20', max(len(unique_segments_ordered), 20)) # Use tab20 for more colors if needed
    segment_colors = {segment: cmap(i % cmap.N) for i, segment in enumerate(unique_segments_ordered)}
else:
    print("Warning: Segment column not properly set up. Using default colors for time series plot.")


if segment_colors: # Only proceed with time series plot if segment_colors was successfully created
    lines_handles = [] # To collect handles for the line legend
    segment_patches = [] # To collect patches for the segment legend

    # Add segment patches to legend list *before* plotting lines for desired legend order
    segment_patches = [Patch(color=segment_colors.get(seg, '#CCCCCC'), alpha=0.3, label=seg) for seg in unique_segments_ordered]


    for i, col in enumerate(SENSOR_COLS):
        # Plot the main time series line
        line_handle, = axes_ts[i].plot(df.index, df[col], label=f'{col} (Raw)', linewidth=1, color='black', zorder=2)
        if i == 0: lines_handles.append(line_handle) # Add only one line handle for the legend

        # Add vertical lines for segment boundaries that are within the data range (excluding min/max)
        # Use the actual bin edges generated by pd.cut
        if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
             # Re-calculate boundaries from the *actual* cut result if possible, or use final_bins
             # Using final_bins is safer as they define the cut points
             segment_boundaries_for_plot = [t for t in final_bins if t > df.index.min() and t < df.index.max()]
             for t in segment_boundaries_for_plot:
                 axes_ts[i].axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=3)

        # Add shaded background colors for segments
        if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
             for k, segment_name in enumerate(unique_segments_ordered): # Iterate through segments with data in order
                 segment_data = df[df['Segment'] == segment_name]
                 if not segment_data.empty:
                     start = segment_data.index.min()
                     end = segment_data.index.max()
                     # Ensure end is > start for plotting span (or handle single points if necessary)
                     if start < end or (start == end and len(segment_data) > 0): # Handle single points for visualization
                          # Ensure end_plot doesn't go beyond the total data range for the axis
                          end_plot = end + pd.Timedelta(seconds=1) if start == end else end
                          end_plot = min(end_plot, df.index.max()) # Cap at max data time

                          axes_ts[i].axvspan(start, end_plot, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=1)


        # Add the Farm_Setup Average line
        if farm_setup_segment_name in segment_colors and farm_setup_avg and col in farm_setup_avg and pd.notna(farm_setup_avg[col]):
             # Find the start and end times of the Farm_Setup segment data subset
             farm_setup_data_subset = df[df['Segment'] == farm_setup_segment_name]
             if not farm_setup_data_subset.empty:
                 farm_setup_start_time = farm_setup_data_subset.index.min()
                 farm_setup_end_time = farm_setup_data_subset.index.max()
                 # Plot the line across the Farm_Setup segment data duration if data exists
                 if pd.notna(farm_setup_start_time) and pd.notna(farm_setup_end_time):
                      # Adjust end time slightly if only one point to make the line visible
                      end_plot_avg = farm_setup_end_time if farm_setup_start_time < farm_setup_end_time else farm_setup_end_time + pd.Timedelta(seconds=1)
                      end_plot_avg = min(end_plot_avg, df.index.max()) # Cap at max data time


                      # Find the corresponding entry in time_definitions for Farm_Setup's START time
                      farm_setup_def_key = None
                      for key, time_val in time_definitions.items():
                           if abs(segment_times.get(key, pd.NaT) - farm_setup_start_time) < pd.Timedelta(seconds=1):
                               farm_setup_def_key = key
                               break

                      label_text = f'{farm_setup_segment_name} Avg'
                      # Optional: Add time range to label if farm_setup_def_key found
                      if farm_setup_def_key and farm_setup_def_key in segment_times and 'hay_start' in segment_times:
                           label_text = f'Baseline ({segment_times[farm_setup_def_key].strftime("%H:%M")} - {segment_times["hay_start"].strftime("%H:%M")}) Avg'
                      elif farm_setup_def_key and farm_setup_def_key in segment_times:
                           # Fallback if hay_start not found but farm_start is
                           label_text = f'Baseline (from {segment_times[farm_setup_def_key].strftime("%H:%M")}) Avg'


                      avg_line_handle, = axes_ts[i].plot([farm_setup_start_time, end_plot_avg],
                                                        [farm_setup_avg[col], farm_setup_avg[col]],
                                                        color='green', linestyle='-', linewidth=2, alpha=0.7, zorder=4,
                                                        label=label_text if i == 0 else "") # Label only on first plot
                      if i == 0: lines_handles.append(avg_line_handle) # Add avg line handle to legend
                 else:
                     print(f"Warning: Farm Setup segment '{farm_setup_segment_name}' data range not suitable for plotting avg line.")
             else:
                 print(f"Warning: No data points found for segment '{farm_setup_segment_name}' to plot average line.")


        axes_ts[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
        axes_ts[i].grid(True, which='major', linestyle=':', linewidth=0.5)
        # Legend will be consolidated at the top

    # Consolidate legends - put one legend at the top
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
    print("Skipping time series plot: segment_colors were not defined or no segments found with data.")


# --- New Chart Comparing Pre-Lab Average to Entire Time Series ---
# Kept this plot as it might still be useful context
print("\n--- Plotting Sensor Data Over Time with Pre-Lab Avg ---")

# Calculate the average for the 'Pre-Lab' segment
prelab_segment_name = df['Segment'].cat.categories[0] if len(df['Segment'].cat.categories) > 0 else None  # Assuming first segment is Pre-Lab
df_prelab = df[df['Segment'] == prelab_segment_name] if prelab_segment_name in df['Segment'].cat.categories else pd.DataFrame()
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


# Only attempt plotting if segment_colors were defined and prelab_avg was calculated
if segment_colors and prelab_segment_name and not df_prelab.empty and prelab_avg:
    fig_prelab, axes_prelab = plt.subplots(nrows=n_sensors, ncols=1, figsize=(18, n_sensors * 4), sharex=True)
    if n_sensors == 1: axes_prelab = [axes_prelab] # Ensure axes_prelab is iterable
    fig_prelab.suptitle('Sensor Readings Over Time Compared to Pre-Lab Average', fontsize=18, y=1.01)

    lines_handles_prelab = [] # To collect handles for the line legend
    segment_patches_prelab = [] # To collect patches for the segment legend

    # Add segment patches to legend list (re-using segment_colors)
    segment_patches_prelab = [Patch(color=segment_colors.get(seg, '#CCCCCC'), alpha=0.3, label=seg) for seg in unique_segments_ordered] # Use unique_segments_ordered


    for i, col in enumerate(SENSOR_COLS):
        # Plot the main time series line
        line_handle, = axes_prelab[i].plot(df.index, df[col], label=f'{col} (Raw)', linewidth=1, color='black', zorder=2)
        if i == 0: lines_handles_prelab.append(line_handle) # Add only one line handle

        # Add segment background colors/lines for context (re-using logic)
        if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
             segment_boundaries_for_plot = [t for t in final_bins if t > df.index.min() and t < df.index.max()]
             for t in segment_boundaries_for_plot:
                 axes_prelab[i].axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=3)

             for k, segment_name in enumerate(unique_segments_ordered): # Iterate through segments with data in order
                 segment_data = df[df['Segment'] == segment_name]
                 if not segment_data.empty:
                     start = segment_data.index.min()
                     end = segment_data.index.max()
                     if start < end or (start == end and len(segment_data) > 0): # Handle single points
                          end_plot = end + pd.Timedelta(seconds=1) if start == end else end
                          end_plot = min(end_plot, df.index.max()) # Cap at max data time
                          axes_prelab[i].axvspan(start, end_plot, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=1)


        # Add the Pre-Lab Average line
        if col in prelab_avg and pd.notna(prelab_avg[col]):
            # Plot the line across the entire data range
            avg_line_handle, = axes_prelab[i].plot([df.index.min(), df.index.max()],
                                                   [prelab_avg[col], prelab_avg[col]],
                                                   color='purple', linestyle='--', linewidth=2, alpha=0.7, zorder=4,
                                                   label=f'{prelab_segment_name} Avg' if i == 0 else "")  # Label only on first plot
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
    print("Skipping Pre-Lab average comparison plots: segment_colors not defined or Pre-Lab segment not found/no data.")


# --- Comparative Distribution Analysis (Box Plots/Violin Plots per Segment) ---
# This plot shows ALL relevant segments for distribution comparison
print("\n--- Comparing Sensor Distributions Across Segments ---")

# Only proceed if segment column is properly set up and there are plottable sensors
if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype) and not df['Segment'].isnull().all():
    n_cols_dist = 2
    # Ensure we only plot for sensors that exist and have data
    plottable_sensors_dist = [col for col in SENSOR_COLS if col in df.columns and not df[col].isnull().all()]
    n_plottable_sensors_dist = len(plottable_sensors_dist)

    if n_plottable_sensors_dist > 0:
        n_rows_dist = (n_plottable_sensors_dist + n_cols_dist - 1) // n_cols_dist
        fig_dist, axes_dist = plt.subplots(n_rows_dist, n_cols_dist, figsize=(16, n_rows_dist * 5), sharex=False) # Don't share x for boxplots usually
        axes_dist = axes_dist.flatten()

        # Get the ordered categories for plotting (using categories that actually have data)
        plot_order_dist = unique_segments_ordered # Use the ordered list with data from the time series plot
        # Create a palette specific to the segments present in the full dataframe that have data
        # Use the same colors as the time series plot for consistency
        palette_dist = {seg: segment_colors.get(seg, '#CCCCCC') for seg in plot_order_dist}


        for i, col in enumerate(plottable_sensors_dist):
            # Use seaborn for easy grouped boxplots or violinplots
            sns.boxplot(x='Segment', y=col, data=df, ax=axes_dist[i],
                        palette=palette_dist, order=plot_order_dist, showfliers=False) # Hide outliers initially
            # Or use violin plot:
            # sns.violinplot(x='Segment', y=col, data=df, ax=axes_dist[i], palette=palette_dist, order=plot_order_dist, inner='quartile')

            axes_dist[i].set_title(f'{col} Distribution by Segment')
            axes_dist[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
            axes_dist[i].set_xlabel('Experimental Segment')
            plt.setp(axes_dist[i].get_xticklabels(), rotation=45, ha="right") # Set rotation and alignment


        # Hide unused subplots
        for j in range(n_plottable_sensors_dist, len(axes_dist)):
            fig_dist.delaxes(axes_dist[j])

        plt.tight_layout()
        plt.savefig('cow_segment_boxplots.png', dpi=300)
        print("Saved segment comparison box plots to 'cow_segment_boxplots.png'")
        # plt.show()
    else:
        print("Skipping distribution plots: No plottable sensor columns found.")
else:
    print("Skipping distribution plots: 'Segment' column not properly set up.")


# --- SPECIFIC NH3 Comparison Boxplot (Baseline vs. Materials) ---
print("\n--- Specific NH3 Comparison: Baseline (Farm Setup) vs. Hay, Cowpatty, Dirt, Grass ---")

# Identify the baseline segment and the comparison segment labels derived from the definition keys
baseline_label = 'Farm_Setup' # Actual label generated
comparison_keys = ['hay_start', 'cowpatty_start', 'dirt_start', 'grass_start']
# Get the actual labels corresponding to these keys based on the segment_label_mapping
comparison_labels = [segment_label_mapping.get(key, key.split('_')[0].capitalize() + ('_to_End' if key == 'grass_start' else '')) for key in comparison_keys]

# Combine the baseline and comparison labels we *want* to plot
target_labels_for_nh3_comparison = [baseline_label] + comparison_labels

# Filter the DataFrame to include only data points from these target labels
try:
    # Create a boolean mask for filtering based on actual generated labels
    mask_nh3_specific_segments = df['Segment'].isin(target_labels_for_nh3_comparison)
    df_nh3_specific = df[mask_nh3_specific_segments].copy()
except TypeError: # Fallback if categories aren't directly comparable (e.g., mixed types, None)
     print("Warning: Error filtering specific segments using isin directly. Falling back to string conversion.")
     df_nh3_specific = df[df['Segment'].astype(str).isin(target_labels_for_nh3_comparison)].copy()
except Exception as e:
     print(f"An unexpected error occurred during specific NH3 segment filtering: {e}")
     df_nh3_specific = pd.DataFrame() # Set to empty to skip plot


# Ensure 'NH3_ppm' column exists and has non-null data within the filtered segments
if 'NH3_ppm' in df_nh3_specific.columns and not df_nh3_specific['NH3_ppm'].isnull().all() and not df_nh3_specific.empty:

    # Determine the *actual* segments present in the filtered data that are also in our target list
    # Use this list for both plotting order and palette creation.
    # We maintain the desired order from target_labels_for_nh3_comparison.
    actual_specific_nh3_segments_present = [label for label in target_labels_for_nh3_comparison if label in df_nh3_specific['Segment'].unique()]

    if actual_specific_nh3_segments_present: # Proceed only if there's data for at least one of the target segments

        # Create a color palette *specifically* for the segments present in this specific plot.
        # Use the colors defined earlier in segment_colors for consistency if possible
        nh3_specific_segment_colors = {seg: segment_colors.get(seg, '#CCCCCC') for seg in actual_specific_nh3_segments_present}
        # If segment_colors was empty or didn't have keys, generate new colors
        if not segment_colors or any(color == '#CCCCCC' for color in nh3_specific_segment_colors.values()):
            print("Warning: Using fallback palette for specific NH3 comparison plot.")
            cmap = plt.get_cmap('tab20', max(len(actual_specific_nh3_segments_present), 20))
            nh3_specific_segment_colors = {segment: cmap(i % cmap.N) for i, segment in enumerate(actual_specific_nh3_segments_present)}


        plt.figure(figsize=(10, 6))
        # Use actual_specific_nh3_segments_present for the 'order' parameter
        # This ensures segments are plotted only if data exists for them, in the order we specified
        sns.boxplot(x='Segment', y='NH3_ppm', data=df_nh3_specific, order=actual_specific_nh3_segments_present, palette=nh3_specific_segment_colors)
        plt.title('NH3 Levels: Baseline (Farm Setup) vs. Materials')
        plt.ylabel('NH3 (ppm)')
        plt.xlabel('Experimental Segment')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
        plt.tight_layout()
        plt.savefig('cow_nh3_baseline_comparison_boxplot.png', dpi=300)
        print("Saved specific NH3 comparison boxplot to 'cow_nh3_baseline_comparison_boxplot.png'")
        # plt.show()

        # Calculate and display statistics for these specific segments
        print("\n--- NH3 Statistics for Baseline vs. Materials ---")
        # Group by the Segment column of the filtered dataframe
        try:
             nh3_stats_specific = df_nh3_specific.groupby('Segment', observed=True)['NH3_ppm'].agg(['mean', 'median', 'std', 'min', 'max'])
        except TypeError: # older pandas might not have observed kwarg
             nh3_stats_specific = df_nh3_specific.groupby('Segment')['NH3_ppm'].agg(['mean', 'median', 'std', 'min', 'max'])
        # Print in the desired order
        print(nh3_stats_specific.loc[actual_specific_nh3_segments_present].dropna(how='all'))


    else:
        print(f"Skipping specific NH3 comparison: No data points were assigned to any of the target segments ({', '.join(target_labels_for_nh3_comparison)}).")

else:
    print("Skipping specific NH3 comparison: Filtered data is empty, 'NH3_ppm' column is missing/all null, or no target segments had data.")


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
        # Identify key segments for comparison - including baseline and comparison segments from NH3 plot
        segments_to_compare_corr = target_labels_for_nh3_comparison # Use the same list as the NH3 plot for consistency
        # Add Pre-farm if it exists and wasn't included
        prefarm_label = df['Segment'].cat.categories[0] if len(df['Segment'].cat.categories) > 0 else None
        if prefarm_label and prefarm_label not in segments_to_compare_corr:
             segments_to_compare_corr.insert(0, prefarm_label) # Add to the beginning

        # Add 'Unknown' if it exists and wasn't included
        if 'Unknown' in df['Segment'].cat.categories and 'Unknown' not in segments_to_compare_corr:
             segments_to_compare_corr.append('Unknown')


        # Filter the list to include only segments that actually exist in the data categories
        actual_segments_to_compare_corr = [seg for seg in segments_to_compare_corr if seg in df['Segment'].cat.categories]

        print(f"\nComparing correlations for segments: {actual_segments_to_compare_corr}")

        for segment_name in actual_segments_to_compare_corr:
            df_segment = df[df['Segment'] == segment_name].copy() # Use .copy()

            # Ensure only numeric columns with variance > 0 within the segment are used
            segment_numeric_cols = df_segment.select_dtypes(include=np.number)

            if not segment_numeric_cols.empty:
                # Drop columns that are constant or NaN within this segment
                segment_numeric_cols = segment_numeric_cols.loc[:, segment_numeric_cols.std() > 0.001] # Check std > small value
                segment_numeric_cols = segment_numeric_cols.dropna(axis=1, how='all') # Drop columns with all NaNs


            if not segment_numeric_cols.empty and len(segment_numeric_cols.columns) > 1:
                 print(f"\n--- Correlation Matrix ({segment_name} Segment) ---")
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
                 print(f"\nSkipping {segment_name} correlation: Not enough numeric columns with variance or data.")

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
    # First, ensure the index is sorted (which it should be, but good practice)
    df_sorted = df[numeric_cols].sort_index()
    # Resampling mean handles gaps by averaging available data in the interval
    df_resampled = df_sorted.resample(resample_freq).mean()

    print(f"\n--- Data Resampled to {resample_freq} Averages ---")
    print(df_resampled.head())

    # Plotting Resampled Data (Example for one sensor)
    sensor_to_plot_resample = 'CO2_ppm'
    if sensor_to_plot_resample in df.columns and sensor_to_plot_resample in df_resampled.columns:
        plt.figure(figsize=(18, 5))
        plt.plot(df_resampled.index, df_resampled[sensor_to_plot_resample], label=f'{sensor_to_plot_resample} ({resample_freq} Avg)', color='red', linewidth=1.5)
        # Only plot raw data if the original column is valid
        if sensor_to_plot_resample in df.columns and not df[sensor_to_plot_resample].isnull().all():
             plt.plot(df.index, df[sensor_to_plot_resample], label=f'{sensor_to_plot_resample} (Raw)', alpha=0.3, color='blue', linewidth=0.5)


        plt.title(f'{sensor_to_plot_resample} Comparison: Raw vs. {resample_freq} Average')
        plt.ylabel(sensor_to_plot_resample.replace('_', ' '))
        # Add segment lines/shading to resampled plot too for context (using segment_colors)
        if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype) and segment_colors:
             # Use the ordered list with data from the time series plot
             for k, segment_name in enumerate(unique_segments_ordered):
                 segment_data = df[df['Segment'] == segment_name]
                 if not segment_data.empty:
                     start = segment_data.index.min()
                     end = segment_data.index.max()
                     if start < end or (start == end and len(segment_data) > 0): # Handle single points
                          end_plot = end + pd.Timedelta(seconds=1) if start == end else end
                          end_plot = min(end_plot, df.index.max()) # Cap at max data time
                          plt.axvspan(start, end_plot, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=1) # Add shading
             # Add vertical boundaries if final_bins is available
             if 'final_bins' in locals():
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
        print(f"Skipping resample plot for {sensor_to_plot_resample}: Column not found or no valid data in original/resampled data.")


    # Rolling Statistics (Example for one sensor)
    # Estimate sample rate (can be inaccurate if gaps exist)
    # Recalculate time diffs on the potentially filtered data
    time_diffs = df.index.to_series().diff().dropna()  # Use original df index for overall rate estimate
    sampling_rate_seconds = time_diffs.median().total_seconds() if not time_diffs.empty else 5  # Default to 5s if no diffs

    if pd.notna(sampling_rate_seconds) and sampling_rate_seconds > 0:
        print(f"Estimated sampling rate for rolling stats: {sampling_rate_seconds:.2f} seconds")
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
            # Check if rolling_mean has data before plotting
            if not rolling_mean.isnull().all():
                 plt.plot(rolling_mean.index, rolling_mean, color='blue', label=f'Rolling Mean ({window_label})', linewidth=1.5)
                 plt.ylabel(sensor_to_roll.replace('_', ' '))
                 plt.title(f'{sensor_to_roll}: Raw Data and Rolling Statistics ({window_label})')

                 # Add Std Dev on secondary axis only if rolling_std has data
                 ax2 = plt.gca().twinx()
                 if not rolling_std.isnull().all():
                     ax2.plot(rolling_std.index, rolling_std, color='red', linestyle='--', label=f'Rolling Std Dev ({window_label})', linewidth=1)
                     ax2.set_ylabel('Rolling Std Dev', color='red')
                     ax2.tick_params(axis='y', labelcolor='red')
                     ax2.spines['right'].set_color('red')

                 # Combine legends
                 lines, labels = plt.gca().get_legend_handles_labels()
                 lines2, labels2 = ax2.get_legend_handles_labels() if 'ax2' in locals() else ([],[])
                 plt.gca().legend(lines + lines2, labels + labels2, loc='upper left')

                 # Add segment lines/shading
                 if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype) and segment_colors:
                      # Use the ordered list with data from the time series plot
                      for k, segment_name in enumerate(unique_segments_ordered):
                          segment_data = df[df['Segment'] == segment_name]
                          if not segment_data.empty:
                              start = segment_data.index.min()
                              end = segment_data.index.max()
                              if start < end or (start == end and len(segment_data) > 0): # Handle single points
                                  end_plot = end + pd.Timedelta(seconds=1) if start == end else end
                                  end_plot = min(end_plot, df.index.max()) # Cap at max data time
                                  plt.axvspan(start, end_plot, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=0) # Use zorder 0 so lines are on top
                      # Add vertical boundaries if final_bins is available
                      if 'final_bins' in locals():
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
                print(f"Skipping rolling stats plot for {sensor_to_roll}: Rolling mean is all null.")

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
    suitable_segments = [s for s in categories_to_consider if not df[df['Segment'] == s].empty]

    if suitable_segments:
        # Pick the segment with the most data points from the suitable ones
        segment_counts = df['Segment'].value_counts()
        # Ensure we only pick from suitable_segments that are actually in the counts and are considered suitable
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
        # Ensure sampling_rate_seconds is defined and positive
        if 'sampling_rate_seconds' in locals() and pd.notna(sampling_rate_seconds) and sampling_rate_seconds > 0:
             samples_per_hour = int(3600 / sampling_rate_seconds)
             samples_per_10_min = int(600 / sampling_rate_seconds)
             # Choose a reasonable period, e.g., 10 minutes or 1 hour, if segment is long enough
             # Ensure period is at least 2 and less than half the data points in the segment
             # Prioritize 10min if possible, otherwise try 1hr/6 (arbitrary fallback)
             potential_periods = [samples_per_10_min, samples_per_hour // 6]
             for p in potential_periods:
                  # Check if the segment is long enough for the period
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
             # Fallback period if sampling rate is unknown or invalid
             decomposition_period = min(max(2, 120), (len(df_segment_decomp) - 1) // 2)  # Fallback: 10-minute period assuming ~12 samples/min (5s rate), ensure valid range

        min_len_for_decomp = 2 * decomposition_period + 1 # Decomposition requires at least 2 periods + 1 point

        print(f"\n--- Attempting Decomposition for '{variable_to_decompose}' in Segment '{segment_to_decompose}' ---")
        print(f"Segment length: {len(df_segment_decomp)} data points.")
        print(f"Selected period: {decomposition_period} samples.")
        print(f"Required minimum length for period {decomposition_period}: {min_len_for_decomp} data points.")


        if len(df_segment_decomp) >= min_len_for_decomp and decomposition_period >= 2:
            try:
                # seasonal_decompose can sometimes handle irregular series, but might issue warnings.
                # Setting freq is ideal but can break if data truly has gaps.
                # Let's try without setting freq first, which is often more robust for real-world data.
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
                print(f"Attempted period: {decomposition_period}. Required length: {min_len_for_decomp}. Actual length: {len(df_segment_decomp)}.")
                print("Try resampling the segment data to a regular frequency before decomposition, or choose a smaller period.")
            except Exception as e:
                 print(f"An unexpected error occurred during decomposition: {e}")
        else:
            print(f"Skipping Decomposition for segment '{segment_to_decompose}': Insufficient data length ({len(df_segment_decomp)}) or invalid period ({decomposition_period}). Required min length for this period: {min_len_for_decomp}.")
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
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
                 segment_times[name] = segment_times[name].tz_localize('UTC').tz_convert(df.index.tzinfo) # Example: Convert via UTC

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


# --- START OF REPLACED SECTION ---

# 7. Add Segment Labels to the DataFrame (The Core Enhancement)
print("\n--- Adding Experimental Segment Labels ---")

# Define the time boundaries and corresponding labels (Keep your original label definition)
time_boundaries = list(segment_times.values()) # Already sorted based on previous step
labels = [
    'Pre-Lab', # Data before lab start
    'Lab',
    'Transition_1', # Time between lab finish and farm start
    'Farm_Setup', # Time between farm start and first material
    'Hay',
    'Cow_Patty',
    'Dirt',
    'Grass_Forage',
    'Post-Grass' # Data after the last defined start time
]

# --- Construct and Clean the Bins ---
# 1. Create the list of potential bin edges
# Use a small offset to ensure min/max are distinct if they coincide with boundaries
min_edge = df.index.min() - pd.Timedelta(microseconds=1)
max_edge = df.index.max() + pd.Timedelta(microseconds=1)
potential_bins = [min_edge] + time_boundaries + [max_edge]

# 2. Sort the list and remove duplicates
final_bins = sorted(list(set(potential_bins)))

# 3. Filter out bins outside the actual data range (optional but cleaner)
# Ensure we only keep bins relevant to the data's timespan
final_bins = [b for b in final_bins if b >= min_edge and b <= max_edge]
# Ensure the absolute start/end are still there after potential filtering/set operation
if not final_bins or final_bins[0] > min_edge:
    final_bins.insert(0, min_edge)
if not final_bins or final_bins[-1] < max_edge:
    final_bins.append(max_edge)
# Re-sort and unique again after potential insertions
final_bins = sorted(list(set(final_bins)))

# Ensure there are at least two bins to define an interval
if len(final_bins) < 2:
    print("Error: Not enough distinct time boundaries to create segments.")
    print("Check segment_times and data time range.")
    # Handle this case appropriately, maybe assign a single label 'Full_Range'
    df['Segment'] = 'Full_Range'
    df['Segment'] = df['Segment'].astype(pd.CategoricalDtype(categories=['Full_Range'], ordered=True))
    print(df['Segment'].value_counts())
    # exit() # Or proceed with caution depending on downstream needs
else:
    # --- Adjust Labels ---
    # The number of labels MUST equal len(final_bins) - 1
    num_intervals = len(final_bins) - 1
    print(f"Number of final bins: {len(final_bins)}")
    # print(f"Final Bins: {final_bins}") # Uncomment for debugging
    print(f"Number of intervals required: {num_intervals}")
    print(f"Original number of labels defined: {len(labels)}")

    # Adjust the labels list to match the number of intervals
    final_labels = labels[:num_intervals] # Take the first 'num_intervals' labels
    # If fewer labels were defined than needed by the bins, pad with generic names
    while len(final_labels) < num_intervals:
        pad_label = f'Unknown_Interval_{len(final_labels)}'
        final_labels.append(pad_label)
        print(f"Warning: Adding generic label '{pad_label}'. Check label definitions against data time range and segment times.")

    print(f"Final number of labels used: {len(final_labels)}")
    # print(f"Final Labels: {final_labels}") # Uncomment for debugging

    # --- Apply pd.cut ---
    # Use include_lowest=True to ensure the very first data point is included
    df['Segment'] = pd.cut(df.index,
                           bins=final_bins,
                           labels=final_labels, # Use the adjusted labels
                           right=False,          # Interval [start, end)
                           ordered=False,        # Treat as categorical initially
                           include_lowest=True) # Include the lowest value in the first bin


    # Handle potential NaNs (though include_lowest=True should minimize this)
    # If any NaNs occur, it means a data point fell outside all defined bins (unlikely with min/max edges)
    if df['Segment'].isnull().any():
        print("Warning: Some data points could not be assigned to a segment. Filling with 'Unknown'.")
        df['Segment'] = df['Segment'].cat.add_categories('Unknown').fillna('Unknown')
    else:
        # Add 'Unknown' category anyway in case it's needed for the ordering list
         if 'Unknown' not in df['Segment'].cat.categories:
              df['Segment'] = df['Segment'].cat.add_categories('Unknown')


    # --- Reorder segment levels (Your existing logic is good) ---
    # Define the desired logical order
    segment_order = [
        'Pre-Lab', 'Lab', 'Transition_1', 'Farm_Setup',
        'Hay', 'Cow_Patty', 'Dirt', 'Grass_Forage', 'Post-Grass'
    ]
    # Add any generated 'Unknown_Interval' labels to the order
    generated_labels = [l for l in final_labels if l.startswith('Unknown_Interval')]
    full_segment_order = segment_order + generated_labels + ['Unknown']

    # Filter order to only include segments actually present in the data
    existing_segments = df['Segment'].unique().tolist()
    # Need to handle potential None/NaN in existing_segments if fillna wasn't perfect
    existing_segments = [seg for seg in existing_segments if pd.notna(seg)]

    ordered_existing_segments = [s for s in full_segment_order if s in existing_segments]

    # Check if all existing segments are in the order, add if missing (e.g., if 'Unknown' occurred)
    for seg in existing_segments:
        if seg not in ordered_existing_segments:
            ordered_existing_segments.append(seg) # Append at the end if missed

    # Apply the final ordering
    df['Segment'] = df['Segment'].astype(pd.CategoricalDtype(categories=ordered_existing_segments, ordered=True))


    print("\nSegment Value Counts:")
    print(df['Segment'].value_counts().sort_index()) # Sort by index for logical order

# --- END OF REPLACED SECTION ---


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
                 print(segment_stats[sensor].dropna(how='all')) # Drop rows where all stats are NaN
             else:
                 print(f"\n--- Skipping Stats for: {sensor} (Not found in segment_stats) ---")
else:
    print("Skipping segment statistics: 'Segment' column not found or not categorical.")


# --- Enhanced Time Series Plotting with Segment Visualization ---
print("\n--- Plotting Sensor Data Over Time with Segments ---")

n_sensors = len(SENSOR_COLS)
fig_ts, axes_ts = plt.subplots(nrows=n_sensors, ncols=1, figsize=(18, n_sensors * 4), sharex=True)
if n_sensors == 1: axes_ts = [axes_ts] # Ensure axes_ts is iterable for single sensor
fig_ts.suptitle('Sensor Readings Over Time by Experimental Segment', fontsize=18, y=1.01)

# Define colors for segments for consistency
if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
    unique_segments = df['Segment'].cat.categories.tolist() # Use ordered categories
    # Use a perceptually uniform colormap like 'viridis' or 'tab10'/'tab20'
    # cmap = plt.get_cmap('viridis', len(unique_segments))
    cmap = plt.get_cmap('tab10', len(unique_segments)) # Tab10 is good for distinct categories
    segment_colors = {segment: cmap(i % cmap.N) for i, segment in enumerate(unique_segments)} # Use modulo for safety if >10 segments

    for i, col in enumerate(SENSOR_COLS):
        # Plot the main time series line
        axes_ts[i].plot(df.index, df[col], label=f'{col} (Raw)', linewidth=1, color='black', zorder=2)

        # Add vertical lines for segment boundaries
        # Use the keys from the sorted segment_times dict for semantic meaning if needed later
        # Plot boundaries that fall within the data range
        segment_start_times_for_plot = [t for name, t in segment_times.items() if t > df.index.min() and t < df.index.max()]
        for t in segment_start_times_for_plot:
            axes_ts[i].axvline(t, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=3)

        # Add shaded background colors for segments (more visual separation)
        # Iterate through the *ordered* unique segments found in the data
        for k, segment_name in enumerate(unique_segments):
            # Find the time range for this segment
            segment_data = df[df['Segment'] == segment_name]
            if not segment_data.empty:
                start = segment_data.index.min()
                end = segment_data.index.max()
                 # Extend end slightly for visual continuity if using axvspan
                end_visual = end + pd.Timedelta(seconds=1) if end < df.index.max() else end

                axes_ts[i].axvspan(start, end_visual, facecolor=segment_colors.get(segment_name, '#CCCCCC'), alpha=0.2, zorder=1, label=f'{segment_name}' if i == 0 else "") # Label only once, provide default color

        # Add annotations for segment starts (optional, can get crowded)
        # for name, t in segment_times.items():
        #     if t > df.index.min() and t < df.index.max():
        #          axes_ts[i].text(t, axes_ts[i].get_ylim()[1] * 0.95, name.replace('_start',''), rotation=90, verticalalignment='top', fontsize=8)

        axes_ts[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
        axes_ts[i].grid(True, which='major', linestyle=':', linewidth=0.5)
        # axes_ts[i].legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Legend outside plot

    # Consolidate legends - put one legend at the top
    handles, labels_plot = axes_ts[0].get_legend_handles_labels() # Get line legend
    # Manually create patches for the segment colors for the main legend
    # Ensure we only create patches for segments that actually exist AND have data
    present_segments = df['Segment'].unique().tolist()
    present_segments = [seg for seg in unique_segments if seg in present_segments] # Keep order
    segment_patches = [Patch(color=segment_colors.get(seg, '#CCCCCC'), alpha=0.3, label=seg) for seg in present_segments]

    fig_ts.legend(handles=handles + segment_patches, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles + segment_patches), 6)) # Adjust ncol

    plt.xlabel('Timestamp')
    # Improve date formatting on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=20)) # Adjust number of ticks
    plt.gcf().autofmt_xdate() # Rotate date labels

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout further if needed
    plt.savefig('cow_timeseries_segmented.png', dpi=300)
    print("Saved segmented time series plots to 'cow_timeseries_segmented.png'")
    # plt.show()

else:
    print("Skipping time series plot: 'Segment' column not properly set up.")


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
        # Check if column exists in df
        if col in df.columns:
            # Use seaborn for easy grouped boxplots or violinplots
            sns.boxplot(x='Segment', y=col, data=df, ax=axes_dist[i],
                        palette=palette, order=plot_order, showfliers=False) # Hide outliers initially
            # Or use violin plot:
            # sns.violinplot(x='Segment', y=col, data=df, ax=axes_dist[i], palette=palette, order=plot_order, inner='quartile')

            axes_dist[i].set_title(f'{col} Distribution by Segment')
            axes_dist[i].set_ylabel(col.replace('_', ' ').replace('pct', '%').replace('F', '°F'))
            axes_dist[i].set_xlabel('Experimental Segment')
            axes_dist[i].tick_params(axis='x', rotation=45)
        else:
            print(f"Skipping distribution plot for {col}: Column not found.")
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
        # Create a combined 'Farm' segment for this analysis if needed
        farm_segments = ['Farm_Setup', 'Hay', 'Cow_Patty', 'Dirt', 'Grass_Forage', 'Post-Grass']
        # Filter farm_segments to those actually present
        farm_segments_present = [s for s in farm_segments if s in df['Segment'].cat.categories]
        df_farm = df[df['Segment'].isin(farm_segments_present)]

        lab_segment_name = 'Lab'
        df_lab = df[df['Segment'] == lab_segment_name] if lab_segment_name in df['Segment'].cat.categories else pd.DataFrame()

        if not df_lab.empty and len(df_lab.index) > 1:
            print(f"\n--- Correlation Matrix ({lab_segment_name} Segment) ---")
            correlation_matrix_lab = df_lab[numeric_cols].corr()
            print(correlation_matrix_lab)
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_lab, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1) # Keep scale consistent
            plt.title(f'Correlation Matrix ({lab_segment_name} Segment)', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'cow_correlation_heatmap_{lab_segment_name}.png')
            print(f"Saved {lab_segment_name} correlation heatmap to cow_correlation_heatmap_{lab_segment_name}.png")
            # plt.show()
        else:
            print(f"\nSkipping {lab_segment_name} correlation: Not enough data or segment not present.")

        if not df_farm.empty and len(df_farm.index) > 1:
            print("\n--- Correlation Matrix (Farm Segments Combined) ---")
            correlation_matrix_farm = df_farm[numeric_cols].corr()
            print(correlation_matrix_farm)
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix_farm, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1) # Keep scale consistent
            plt.title('Correlation Matrix (Farm Segments Combined)', fontsize=16)
            plt.tight_layout()
            plt.savefig('cow_correlation_heatmap_farm.png')
            print("Saved farm correlation heatmap to 'cow_correlation_heatmap_farm.png'")
            # plt.show()
        else:
            print("\nSkipping Farm correlation: Not enough data or segments not present.")
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
    df_resampled = df[numeric_cols].resample(resample_freq).mean()

    print(f"\n--- Data Resampled to {resample_freq} Averages ---")
    print(df_resampled.head())

    # Plotting Resampled Data (Example for one sensor)
    sensor_to_plot = 'CO2_ppm'
    if sensor_to_plot in df.columns and sensor_to_plot in df_resampled.columns:
        plt.figure(figsize=(18, 5))
        plt.plot(df_resampled.index, df_resampled[sensor_to_plot], label=f'{sensor_to_plot} ({resample_freq} Avg)', color='red', linewidth=1.5)
        plt.plot(df.index, df[sensor_to_plot], label=f'{sensor_to_plot} (Raw)', alpha=0.3, color='blue', linewidth=0.5)
        plt.title(f'{sensor_to_plot} Comparison: Raw vs. {resample_freq} Average')
        plt.ylabel(sensor_to_plot.replace('_', ' '))
        # Add segment lines/shading to resampled plot too for context
        # Get boundaries again if needed
        segment_start_times_for_plot = [t for name, t in segment_times.items() if t > df.index.min() and t < df.index.max()]
        for t in segment_start_times_for_plot:
             plt.axvline(t, color='grey', linestyle=':', linewidth=1, alpha=0.7)
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(f'cow_{sensor_to_plot}_resample_comparison_{resample_freq}.png')
        print(f"Saved {sensor_to_plot} resample comparison plot.")
        # plt.show()
    else:
        print(f"Skipping resample plot for {sensor_to_plot}: Column not found in original or resampled data.")


    # Rolling Statistics (Example for one sensor)
    # Estimate sample rate (can be inaccurate if gaps exist)
    time_diffs = df.index.to_series().diff().median()
    if pd.notna(time_diffs):
        sampling_rate_seconds = time_diffs.total_seconds()
        print(f"Estimated sampling rate: {sampling_rate_seconds:.2f} seconds")
        window_size_minutes = 5 # 5 minutes
        # Calculate window size in samples (handle potential zero sampling rate)
        window_size_samples = max(1, int((window_size_minutes * 60) / sampling_rate_seconds)) if sampling_rate_seconds > 0 else 10 # Default if rate unknown
        window_label = f'{window_size_minutes}min'

        print(f"\n--- Calculating Rolling Statistics (Window: {window_label} / {window_size_samples} samples) ---")
        sensor_to_roll = 'NH3_ppm'

        if sensor_to_roll in df.columns:
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
            for t in segment_start_times_for_plot:
                 plt.axvline(t, color='grey', linestyle=':', linewidth=1, alpha=0.7)
            plt.grid(True, axis='y', linestyle=':')
            plt.xlabel('Timestamp')
            plt.tight_layout()
            plt.savefig(f'cow_{sensor_to_roll}_rolling_stats_{window_label}.png')
            print(f"Saved {sensor_to_roll} rolling stats plot.")
            # plt.show()
        else:
            print(f"Skipping rolling stats plot for {sensor_to_roll}: Column not found.")
    else:
        print("Skipping rolling stats: Could not determine sampling rate.")



# --- Time Series Decomposition (Consider decomposing key segments) ---
# Decomposing the *entire* series might mix different behaviors.
# Consider decomposing a long, relatively stable segment like 'Lab' or a 'Farm' period if long enough.

if 'Segment' in df.columns and isinstance(df['Segment'].dtype, pd.CategoricalDtype):
    variable_to_decompose = 'Temperature_F'
    segment_to_decompose = 'Lab' # Example: Decompose only the Lab segment

    if variable_to_decompose in df.columns and segment_to_decompose in df['Segment'].cat.categories:
        df_segment_decomp = df[df['Segment'] == segment_to_decompose][variable_to_decompose].dropna()

        # Choose period carefully. Based on estimated sampling rate
        if pd.notna(time_diffs) and sampling_rate_seconds > 0:
             samples_per_hour = int(3600 / sampling_rate_seconds)
             samples_per_10_min = int(600 / sampling_rate_seconds)
             # Choose a reasonable period, e.g., 10 minutes or 1 hour, if segment is long enough
             decomposition_period = max(2, samples_per_10_min if samples_per_10_min > 1 else samples_per_hour // 6) # Ensure period >= 2
        else:
             decomposition_period = 12 * 10 # Fallback: 10-minute period assuming 12 samples/min (5s rate)

        min_len_for_decomp = 2 * decomposition_period

        print(f"\n--- Attempting Decomposition for '{variable_to_decompose}' in Segment '{segment_to_decompose}' ---")
        print(f"Segment length: {len(df_segment_decomp)} data points.")
        print(f"Selected period: {decomposition_period} samples.")
        print(f"Required length for period {decomposition_period}: {min_len_for_decomp} data points.")

        if len(df_segment_decomp) >= min_len_for_decomp:
            try:
                # Ensure the index has a frequency if possible, or handle decomposition without it
                # df_segment_decomp = df_segment_decomp.asfreq(pd.Timedelta(seconds=sampling_rate_seconds)) # Might introduce NaNs
                decomposition = seasonal_decompose(df_segment_decomp, model='additive', period=decomposition_period)
                fig_decomp = decomposition.plot()
                fig_decomp.set_size_inches(14, 10)
                fig_decomp.suptitle(f'Time Series Decomposition of {variable_to_decompose} (Segment: {segment_to_decompose}, Period: {decomposition_period})', y=1.03)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                plt.savefig(f'cow_decomposition_{variable_to_decompose}_{segment_to_decompose}.png')
                print(f"Saved decomposition plot to 'cow_decomposition_{variable_to_decompose}_{segment_to_decompose}.png'")
                # plt.show()
            except ValueError as e:
                print(f"Could not perform decomposition for segment '{segment_to_decompose}', Error: {e}")
                print("This might be due to insufficient length, NaNs, or irregular time intervals within the segment.")
            except Exception as e:
                 print(f"An unexpected error occurred during decomposition: {e}")
        else:
            print(f"Skipping Decomposition for segment '{segment_to_decompose}': Insufficient data length for the chosen period.")
    else:
        print(f"Skipping decomposition: Variable '{variable_to_decompose}' or segment '{segment_to_decompose}' not found.")
else:
    print("Skipping decomposition: 'Segment' column not available.")


# --- Potential Next Steps & Advanced Analysis ---
print("\n--- Analysis Complete ---")
print("Consider further analysis:")
print("1.  Statistical Tests: Use ANOVA or t-tests (or non-parametric equivalents like Kruskal-Wallis/Mann-Whitney U) to formally compare distributions between key segments (e.g., Lab vs. Hay vs. Grass for NH3_ppm).")
print("2.  Outlier Investigation: Analyze the outliers potentially hidden in box plots (re-run with showfliers=True). Are they errors or significant events?")
print("3.  Change Point Detection: Use algorithms (e.g., from `ruptures` library) to automatically detect shifts in sensor behavior and compare with defined segments.")
print("4.  Feature Engineering: Create new features (e.g., rates of change `df[col].diff()`, ratios between gases `df['NH3_ppm'] / df['CO2_ppm']`).")
print("5.  Environmental Correlation: If available, correlate sensor data with external factors (e.g., ambient temperature, ventilation changes, animal activity logs).")
print("6.  Interactive Visualizations: Use libraries like Plotly or Bokeh for plots you can zoom, pan, and hover over for details.")
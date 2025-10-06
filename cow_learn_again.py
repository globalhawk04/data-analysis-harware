import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.dates as mdates
import numpy as np # Need numpy for array creation
# Import statsmodels for time series modeling
from statsmodels.tsa.arima.model import ARIMA
# Import pmdarima for automatic ARIMA order selection (Optional but helpful)
# If you don't have pmdarima, you'll need to manually specify the ARIMA order
try:
    from pmdarima import auto_arima
    use_auto_arima = True
    print("pmdarima found. Using auto_arima for initial model order selection.")
except ImportError:
    use_auto_arima = False
    print("pmdarima not found. Cannot use auto_arima. You may need to install it (`pip install pmdarima`) or manually specify ARIMA orders.")
    print("Falling back to a simple default ARIMA(1,0,0) model if auto_arima is needed.")


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
coffee_start = '2025-05-06 13:22:57' # Start of the plotting and analysis range
coffee_end = '2025-05-06 16:43:42' # End of the plotting and analysis range

# Convert timestamps to datetime objects
segmented_at = pd.to_datetime(segmented_at_str)
coffee_start_dt = pd.to_datetime(coffee_start)
coffee_end_dt = pd.to_datetime(coffee_end)
farm_start_dt = pd.to_datetime(farm_start)
farm_base_line_1_dt = pd.to_datetime(farm_base_line_1)


# --- Step 1 & 2: Calculate Baseline Averages ---
# Using the specified baseline period
baseline_df = df[farm_start_dt : farm_base_line_1_dt]

baseline_averages = None # Initialize baseline_averages
if baseline_df.empty:
    print(f"Warning: No data found in the baseline period ({farm_start} to {farm_base_line_1}). Cannot calculate baseline average.")
else:
    baseline_averages = baseline_df.mean()
    print(f"\nBaseline Averages ({farm_start} to {farm_base_line_1}):")
    print(baseline_averages)
    print("-" * 40)


# --- Step 3: Define the data to be plotted and analyzed (Comparison Period - Coffee) ---
# Select the data range for the coffee phase
# Ensure the segmented_at point is included if it falls within the range boundaries
filtered_df = df[coffee_start_dt : coffee_end_dt].copy()

if filtered_df.empty:
    print(f"Warning: No data found in the plotting period ({coffee_start} to {coffee_end}). Cannot plot or analyze.")
    exit()

# Also check if the segmentation point is within or adjacent to the plotting range
# We need data BEFORE the point to establish baseline behavior within the analysis range
# and data AFTER the point to see the effect.
if segmented_at < filtered_df.index.min() or segmented_at >= filtered_df.index.max():
    print(f"Warning: Segmentation point ({segmented_at_str}) is outside or at the very end of the plotting data range.")
    print("Cannot perform time series intervention analysis as there is not enough data before/after the point.")
    perform_segmentation = False
    perform_analysis = False
else:
     print(f"Plotting and Analysis Data Range: {coffee_start} to {coffee_end}")
     print(f"Number of data points to plot/analyze: {len(filtered_df)}")
     print(f"Segmenting plot and analyzing change at: {segmented_at_str}")
     print("-" * 40)
     perform_segmentation = True
     perform_analysis = True # Set to True if conditions met


# --- Prepare for Analysis: Create the Intervention Variable ---
# This variable will be 0 before or at segmented_at, and 1 after segmented_at
if perform_analysis:
    # The intervention variable must align with the index of filtered_df
    intervention_variable = (filtered_df.index > segmented_at).astype(int)
    # Convert to DataFrame if needed by statsmodels (sometimes helpful for named columns)
    intervention_variable = pd.DataFrame(intervention_variable, index=filtered_df.index, columns=['intervention_step'])


# --- Plotting ---
# Ensure we only plot columns that actually exist in filtered_df
cols_to_plot = filtered_df.columns.tolist() # Get list of column names

if len(cols_to_plot) == 0:
    print("No columns available to plot/analyze in filtered_df.")
    exit()

# Create the figure and subplots
fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=1, figsize=(12, len(cols_to_plot)*3.5), sharex=True) # Increased figsize height slightly

fig.suptitle('Sensor Readings with Segmentation and Intervention Analysis', fontsize=16)

# Ensure axes is iterable even if only one column
if len(cols_to_plot) == 1:
    axes = [axes]


print("\n--- Intervention Analysis Results ---")
analysis_results = {} # Dictionary to store results for printing/annotation

for i, col in enumerate(cols_to_plot):
    ax = axes[i] # Get the current axis for this column
    y = filtered_df[col] # Data for the current column

    # --- Perform Intervention Analysis ---
    analysis_success = False
    if perform_analysis:
        try:
            # Find baseline model order using auto_arima on data BEFORE intervention (within filtered_df range)
            baseline_data_for_arima = y[y.index <= segmented_at]

            if use_auto_arima and len(baseline_data_for_arima) > 10: # Need enough data for auto_arima
                 # Use a subset of standard d values and suppress warnings for cleaner output
                 arima_model_fit_baseline = auto_arima(
                     baseline_data_for_arima,
                     seasonal=False, # Assuming no strong seasonality within a few hours
                     stepwise=True,
                     suppress_warnings=True,
                     error_action='ignore',
                     trace=False, # Set to True to see auto_arima progress
                     max_p=5, max_q=5, # Limit complexity
                     d=None # Let auto_arima determine d
                 )
                 order = arima_model_fit_baseline.order
                 print(f"Auto-ARIMA order for {col} baseline: {order}")
            elif len(baseline_data_for_arima) > 0: # Fallback or if pmdarima not used
                # Default to a simple ARIMA(1,0,0) if auto_arima can't be used or not enough data
                order = (1,0,0)
                print(f"Using default ARIMA order {order} for {col} (auto_arima skipped or not enough data).")
            else:
                 print(f"Not enough baseline data for ARIMA modeling for {col}.")
                 order = None


            if order is not None and len(y) > max(order) * 2: # Need sufficient total data points
                 # Fit ARIMA model with the intervention variable as exogenous
                 # Use method='innovations_mle' or 'statespace' for robustness with exogenous variables
                 model = ARIMA(y, exog=intervention_variable, order=order)
                 model_fit = model.fit()

                 # Find the parameter for the intervention variable (usually named 'x1' or based on column name)
                 # Check the results summary params to find the exact name
                 intervention_param_name = None
                 for param_name in model_fit.params.index:
                     if 'intervention_step' in param_name or 'x1' in param_name: # Look for names likely assigned to the exog var
                         intervention_param_name = param_name
                         break

                 if intervention_param_name:
                    estimated_change = model_fit.params[intervention_param_name]
                    p_value = model_fit.pvalues[intervention_param_name]

                    # Store results
                    analysis_results[col] = {
                        'estimated_change': estimated_change,
                        'p_value': p_value,
                        'order': order # Store the ARIMA order used
                    }
                    print(f"  {col}: Estimated Change = {estimated_change:.4f}, p-value = {p_value:.4f}")
                    analysis_success = True
                 else:
                     print(f"  Could not find intervention variable parameter in results for {col}.")


            else:
                 print(f"  Not enough total data points or baseline data for ARIMA fitting for {col}.")


        except Exception as e:
            print(f"  Analysis failed for {col}: {e}")

    if perform_segmentation:
        # --- Segment the data for the current column based on time ---
        data_before = filtered_df[filtered_df.index <= segmented_at]
        data_after = filtered_df[filtered_df.index > segmented_at]

        # --- Plot the segmented data ---
        # Plot the data BEFORE the segmentation point (e.g., blue)
        if not data_before.empty:
            ax.plot(data_before.index, data_before[col], color='blue', linestyle='-', label=f'{col} (Before)')

        # Plot the data AFTER the segmentation point (e.g., green)
        if not data_after.empty:
             # Ensure the line connects by potentially including the last point before
             connecting_point = filtered_df[filtered_df.index <= segmented_at].tail(1)
             if not connecting_point.empty:
                 # Check if the connecting point is not already the first point of data_after
                 if data_after.empty or connecting_point.index[0] != data_after.index[0]:
                      combined_after = pd.concat([connecting_point, data_after])
                 else: # The last point before is the same as the first point after
                      combined_after = data_after
             else:
                 combined_after = data_after

             # Only plot if combined_after is not empty
             if not combined_after.empty:
                ax.plot(combined_after.index, combined_after[col], color='green', linestyle='-', label=f'{col} (After)')

        # --- Add a vertical line marker at the segmentation point ---
        ax.axvline(x=segmented_at, color='gray', linestyle=':', linewidth=1, label='_nolegend_')

    else: # Plot as a single line if segmentation is not applicable
         ax.plot(filtered_df.index, filtered_df[col], label=col)


    # --- Plot the Baseline Average Line ---
    # Check if baseline was calculated and average exists for this column
    if baseline_averages is not None and col in baseline_averages.index:
         # Plot a horizontal line at the baseline average for this column
         ax.axhline(y=baseline_averages[col], color='red', linestyle='--', label='Baseline Avg' if i == 0 else "_nolegend_")


    ax.set_ylabel(col)
    ax.grid(True) # Add grid for better readability


    # --- Add Analysis Results Annotation to Plot ---
    if col in analysis_results:
        res = analysis_results[col]
        # Determine significance stars
        stars = ''
        if res['p_value'] < 0.001:
            stars = '***'
        elif res['p_value'] < 0.01:
            stars = '**'
        elif res['p_value'] < 0.05:
            stars = '*'
        elif res['p_value'] < 0.1:
             stars = '.' # Typically used for p < 0.1

        annotation_text = f"Change Est: {res['estimated_change']:.3f}\np={res['p_value']:.3f}{stars}"
        # Position the text (e.g., upper right corner of the subplot)
        ax.text(0.98, 0.98, annotation_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # Add legend after annotations
    ax.legend(loc='upper left')


# --- Configure the X-axis for date/time display ---
# Apply this to the bottom-most axis
ax_bottom = axes[-1]

# Set the major locator (e.g., every 10 minutes for clarity over a few hours)
ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

# Set the major formatter to display HH:MM
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Optional: Set minor ticks for finer granularity (e.g., every 1 minute)
ax_bottom.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))


# Improve the layout and prevent labels overlapping
plt.xlabel('Timestamp') # Still set the overall xlabel on the bottom axis
fig.autofmt_xdate() # Auto-format the x-axis labels (rotate them)
plt.tight_layout(rect=[0,0.03,1,0.96]) # Adjust layout to make room for suptitle and annotations


# Save the figure
plt.savefig('cow_learn_3_analysis.png') # Changed filename
#plt.show() # Display the plot
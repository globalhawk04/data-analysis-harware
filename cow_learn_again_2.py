import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.dates as mdates
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
try:
    from pmdarima import auto_arima
    use_auto_arima = True
    print("pmdarima found. Using auto_arima for initial model order selection.")
except ImportError:
    use_auto_arima = False
    print("pmdarima not found. Cannot use auto_arima. Install with `pip install pmdarima` or manually specify ARIMA orders.")
    print("Falling back to a simple default ARIMA(1,0,0) model if auto_arima is needed.")

# --- Function to Identify and Impute Outliers ---
# This function takes a pandas Series (one sensor reading over time)
# and identifies potential spikes based on deviation from a rolling median.
def identify_and_impute_outliers(series, window_size=7, threshold_factor=3.5):
    """
    Identifies and imputes outlier spikes in a time series using rolling statistics.

    Args:
        series (pd.Series): The input time series data for a single sensor.
        window_size (int): The size of the rolling window. Should be odd (e.g., 5, 7, 9).
        threshold_factor (float): Number of rolling Median Absolute Deviations (MAD)
                                  away from the rolling median to consider a point an outlier.
                                  3.5 is a common value (approx equivalent to 3.5 std devs).

    Returns:
        pd.Series: The series with outliers replaced by interpolated values.
    """
    # Ensure series is sorted by index (timestamp)
    series = series.sort_index()

    # Calculate rolling median (center window helps capture local trend)
    # min_periods ensures we have enough data in the window
    rolling_median = series.rolling(window=window_size, center=True, min_periods=1).median()

    # Calculate rolling Median Absolute Deviation (MAD)
    # MAD is more robust to outliers than standard deviation
    rolling_mad = series.rolling(window=window_size, center=True, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)

    # Handle potential division by zero if rolling_mad is 0
    # Replace 0 rolling_mad with a very small number or the global MAD if needed
    global_mad = np.median(np.abs(series - np.median(series)))
    rolling_mad = rolling_mad.replace(0, global_mad if global_mad > 0 else 1e-9) # Use global MAD if non-zero, otherwise a tiny value


    # Define the outlier threshold based on scaled rolling MAD
    # 0.6745 is the scaling factor for MAD to be equivalent to STD in a normal distribution
    threshold = threshold_factor * rolling_mad / 0.6745 # Scale MAD to be comparable to STD

    # Identify outliers: points that are far from the rolling median
    outlier_indices = series.index[np.abs(series - rolling_median) > threshold]

    print(f"  Identified {len(outlier_indices)} potential outliers in this series.")
    # Optional: print outlier timestamps and values if you want to review them
    # if len(outlier_indices) > 0:
    #    print("    Outlier timestamps:")
    #    for ts in outlier_indices:
    #        print(f"      {ts}: {series.loc[ts]}")


    # Create a copy to avoid modifying the original series outside this function
    series_cleaned = series.copy()

    # Replace identified outliers with NaN
    series_cleaned.loc[outlier_indices] = np.nan

    # Impute the NaN values using linear interpolation
    # This fills the gaps created by removing outliers
    series_cleaned = series_cleaned.interpolate(method='linear')

    # Handle potential NaNs at the start/end if they couldn't be interpolated
    series_cleaned = series_cleaned.fillna(method='bfill').fillna(method='ffill')


    return series_cleaned

# --- End of Outlier Handling Function ---


# Load data and initial processing
df = pd.read_csv('sensor_readings_export.csv')
df = df.drop(columns=['ID'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp').reset_index(drop=True)
df = df.set_index('Timestamp')

# Keep relevant time ranges defined
segmented_at_str = '2025-05-06 16:00:00' # String for the segmentation point
coffee_start = '2025-05-06 15:12:06' # Start of the plotting and analysis range
coffee_end = '2025-05-06 16:43:42' # End of the plotting and analysis range


# Convert timestamps to datetime objects
segmented_at = pd.to_datetime(segmented_at_str)
coffee_start_dt = pd.to_datetime(coffee_start)
coffee_end_dt = pd.to_datetime(coffee_end)

# --- Define the data to be plotted and analyzed ---
filtered_df = df[coffee_start_dt : coffee_end_dt].copy()

if filtered_df.empty:
    print(f"Warning: No data found in the plotting period ({coffee_start} to {coffee_end}). Cannot plot or analyze.")
    exit()

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
     perform_analysis = True

# --- Apply Outlier Handling to Create the Cleaned Dataset ---
# This is the new preprocessing step
print("\n--- Applying Outlier Detection and Imputation ---")
filtered_df_cleaned = filtered_df.copy() # Create a copy to clean
if perform_analysis: # Only clean if we're going to analyze
    for col in filtered_df_cleaned.columns:
        print(f"Processing column: {col}")
        filtered_df_cleaned[col] = identify_and_impute_outliers(
            filtered_df_cleaned[col],
            window_size=7, # You can adjust window_size and threshold_factor
            threshold_factor=3.5
        )
print("-" * 40)


# --- Prepare for Analysis: Create the Intervention Variable ---
if perform_analysis:
    intervention_variable = (filtered_df.index > segmented_at).astype(int)
    intervention_variable = pd.DataFrame(intervention_variable, index=filtered_df.index, columns=['intervention_step'])


# --- Plotting ---
cols_to_plot = filtered_df.columns.tolist() # Get list of column names

if len(cols_to_plot) == 0:
    print("No columns available to plot/analyze in filtered_df.")
    exit()

fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=1, figsize=(12, len(cols_to_plot)*3.5), sharex=True)
fig.suptitle('Sensor Readings with Segmentation and Intervention Analysis (Original vs. Cleaned)', fontsize=16)

if len(cols_to_plot) == 1:
    axes = [axes]

print("\n--- Intervention Analysis Results ---")
analysis_results = {}

for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    y_original = filtered_df[col] # Original data
    y_cleaned = filtered_df_cleaned[col] # Cleaned data

    analysis_results[col] = {'original': None, 'cleaned': None, 'error': None}

    # --- Perform Intervention Analysis on Original Data ---
    if perform_analysis:
        try:
            baseline_data_for_arima_original = y_original[y_original.index <= segmented_at]
            order_original = (1,0,0) # Default

            if use_auto_arima and len(baseline_data_for_arima_original) > 10:
                 arima_model_fit_baseline = auto_arima(baseline_data_for_arima_original, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False, max_p=5, max_q=5, d=None)
                 order_original = arima_model_fit_baseline.order
                 # print(f"  Original - Auto-ARIMA order for {col}: {order_original}") # Optional: print auto-arima order for original

            if order_original is not None and len(y_original) > max(order_original or [0]) * 2: # Ensure max([0]) for safety
                 model_original = ARIMA(y_original, exog=intervention_variable, order=order_original)
                 model_fit_original = model_original.fit()

                 intervention_param_name_original = None
                 for param_name in model_fit_original.params.index:
                     if 'intervention_step' in param_name or 'x1' in param_name:
                         intervention_param_name_original = param_name
                         break

                 if intervention_param_name_original:
                    analysis_results[col]['original'] = {
                        'estimated_change': model_fit_original.params[intervention_param_name_original],
                        'p_value': model_fit_original.pvalues[intervention_param_name_original],
                        'order': order_original
                    }
                    print(f"  {col} (Original): Estimated Change = {analysis_results[col]['original']['estimated_change']:.4f}, p-value = {analysis_results[col]['original']['p_value']:.4f}")

            else:
                print(f"  {col} (Original): Not enough data for ARIMA fitting.")


        except Exception as e:
            print(f"  {col} (Original): Analysis failed: {e}")
            analysis_results[col]['original'] = {'error': str(e)} # Store error


    # --- Perform Intervention Analysis on Cleaned Data ---
    if perform_analysis:
        try:
            baseline_data_for_arima_cleaned = y_cleaned[y_cleaned.index <= segmented_at]
            order_cleaned = (1,0,0) # Default

            if use_auto_arima and len(baseline_data_for_arima_cleaned) > 10:
                 arima_model_fit_baseline = auto_arima(baseline_data_for_arima_cleaned, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False, max_p=5, max_q=5, d=None)
                 order_cleaned = arima_model_fit_baseline.order
                 # print(f"  Cleaned - Auto-ARIMA order for {col}: {order_cleaned}") # Optional: print auto-arima order for cleaned


            # Important: Use the *same* intervention_variable dataframe index, as it aligns with y_cleaned
            if order_cleaned is not None and len(y_cleaned) > max(order_cleaned or [0]) * 2:
                 model_cleaned = ARIMA(y_cleaned, exog=intervention_variable, order=order_cleaned)
                 model_fit_cleaned = model_cleaned.fit()

                 intervention_param_name_cleaned = None
                 for param_name in model_fit_cleaned.params.index:
                     if 'intervention_step' in param_name or 'x1' in param_name:
                         intervention_param_name_cleaned = param_name
                         break

                 if intervention_param_name_cleaned:
                    analysis_results[col]['cleaned'] = {
                        'estimated_change': model_fit_cleaned.params[intervention_param_name_cleaned],
                        'p_value': model_fit_cleaned.pvalues[intervention_param_name_cleaned],
                        'order': order_cleaned
                    }
                    print(f"  {col} (Cleaned): Estimated Change = {analysis_results[col]['cleaned']['estimated_change']:.4f}, p-value = {analysis_results[col]['cleaned']['p_value']:.4f}")

            else:
                 print(f"  {col} (Cleaned): Not enough data for ARIMA fitting.")


        except Exception as e:
            print(f"  {col} (Cleaned): Analysis failed: {e}")
            analysis_results[col]['cleaned'] = {'error': str(e)} # Store error


    # --- Plotting the Cleaned Data with Segmentation ---
    if perform_segmentation: # Use cleaned data for plotting
        data_before = y_cleaned[y_cleaned.index <= segmented_at]
        data_after = y_cleaned[y_cleaned.index > segmented_at]

        if not data_before.empty:
            ax.plot(data_before.index, data_before, color='blue', linestyle='-', label=f'{col} (Before)')

        if not data_after.empty:
             connecting_point = y_cleaned[y_cleaned.index <= segmented_at].tail(1)
             if not connecting_point.empty:
                 if data_after.empty or connecting_point.index[0] != data_after.index[0]:
                      combined_after = pd.concat([connecting_point, data_after])
                 else:
                      combined_after = data_after
             else:
                 combined_after = data_after

             if not combined_after.empty:
                ax.plot(combined_after.index, combined_after, color='green', linestyle='-', label=f'{col} (After)')

        ax.axvline(x=segmented_at, color='gray', linestyle=':', linewidth=1, label='_nolegend_')
    else:
         ax.plot(y_cleaned.index, y_cleaned, label=col) # Plot cleaned data as single line


    ax.set_ylabel(col)
    ax.grid(True)


    # --- Add Analysis Results Annotation to Plot (Both Original and Cleaned) ---
    annotation_text_lines = []

    # Original Results
    if analysis_results[col]['original'] and 'error' not in analysis_results[col]['original']:
        res = analysis_results[col]['original']
        stars = ''
        if res['p_value'] < 0.001: stars = '***'
        elif res['p_value'] < 0.01: stars = '**'
        elif res['p_value'] < 0.05: stars = '*'
        elif res['p_value'] < 0.1: stars = '.'
        annotation_text_lines.append(f"Orig: Est={res['estimated_change']:.3f}, p={res['p_value']:.3f}{stars}")
    elif analysis_results[col]['original'] and 'error' in analysis_results[col]['original']:
         annotation_text_lines.append(f"Orig Error: {analysis_results[col]['original']['error'][:30]}...") # Truncate error


    # Cleaned Results
    if analysis_results[col]['cleaned'] and 'error' not in analysis_results[col]['cleaned']:
        res = analysis_results[col]['cleaned']
        stars = ''
        if res['p_value'] < 0.001: stars = '***'
        elif res['p_value'] < 0.01: stars = '**'
        elif res['p_value'] < 0.05: stars = '*'
        elif res['p_value'] < 0.1: stars = '.'
        annotation_text_lines.append(f"Clean: Est={res['estimated_change']:.3f}, p={res['p_value']:.3f}{stars}")
    elif analysis_results[col]['cleaned'] and 'error' in analysis_results[col]['cleaned']:
        annotation_text_lines.append(f"Clean Error: {analysis_results[col]['cleaned']['error'][:30]}...") # Truncate error

    if annotation_text_lines:
        annotation_text = "\n".join(annotation_text_lines)
        ax.text(0.98, 0.98, annotation_text, transform=ax.transAxes, fontsize=8, # Reduced fontsize slightly
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))


    ax.legend(loc='upper left')


# --- Configure the X-axis for date/time display ---
ax_bottom = axes[-1]
ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_bottom.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))


# Improve the layout
plt.xlabel('Timestamp')
fig.autofmt_xdate()
plt.tight_layout(rect=[0,0.03,1,0.96]) # Adjust rect for suptitle/annotations


# Save the figure
plt.savefig('cow_learn_5_analysis_cleaned.png')
#plt.show()
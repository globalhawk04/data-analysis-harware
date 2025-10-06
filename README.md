This collection of scripts represents a sophisticated, iterative journey of data exploration, cleaning, and statistical analysis. 
Time-Series Analysis of Livestock Environmental Sensor Data

![alt text](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![alt text](https://img.shields.io/badge/Pandas-2.0-blue?logo=pandas)
![alt text](https://img.shields.io/badge/Matplotlib-3.7-blue?logo=matplotlib)
![alt text](https://img.shields.io/badge/Statsmodels-0.14-blue)

This repository contains a suite of Python scripts for the comprehensive analysis of IoT sensor data collected from a livestock environment. The project showcases an end-to-end data science workflow, from initial exploratory data analysis (EDA) to advanced statistical modeling, designed to extract meaningful insights from noisy, real-world time-series data.

The core objective is to analyze the impact of various environmental interventions (e.g., introducing different materials like hay, cow patties) on atmospheric gas levels (NH3, CO2, etc.) and to quantify these changes using statistical methods.

Project Workflow & Script Evolution

The analysis was conducted in an iterative manner, with each script building upon the insights and addressing the limitations of the previous one.

1. Initial Exploration (cow_it_*.py series)

Purpose: Initial data loading, cleaning, and high-level Exploratory Data Analysis (EDA).

Key Activities:

Loading raw CSV data and parsing timestamps (cow_it.py).

Generating descriptive statistics, distribution plots (histograms, box plots), and initial time-series visualizations (cow_it_1.py).

Calculating overall correlation matrices and creating pair plots to understand relationships between sensors (cow_it_2.py).

Experimenting with resampling and rolling statistics to smooth the data and identify trends (cow_it_3.py).

2. Experimental Segmentation (cow_learn_segments*.py series)

Purpose: To label the time-series data according to the specific real-world experimental phases (e.g., "Baseline," "Hay Test," "Cowpatty Test").

Key Activities:

Slicing the DataFrame into distinct time-based segments.

Visualizing sensor readings with colored backgrounds to clearly delineate each experimental phase (cow_learn_segments.py).

Generating separate correlation matrices for each segment to analyze how sensor relationships change under different conditions (cow_learn_segments_correlate.py).

3. Statistical Intervention Analysis (cow_learn_again*.py series)

Purpose: This is the most advanced stage of the analysis. The goal is to move beyond visual inspection and statistically quantify the impact of an intervention.

Key Activities:

Intervention Modeling: Implements an ARIMA (AutoRegressive Integrated Moving Average) model with an exogenous variable (the "intervention") to estimate the magnitude and statistical significance (p-value) of the change in sensor readings after a specific event.

Automated Model Selection: Uses pmdarima's auto_arima to find the optimal order for the baseline time-series model.

Outlier Detection & Imputation: Introduces a robust outlier detection function using a rolling median and Median Absolute Deviation (MAD) to clean noisy data spikes that could otherwise skew the statistical results (cow_learn_again_2.py, cow_learn_again_3.py).

Comparative Analysis: The final scripts run the ARIMA intervention analysis on both the original and the cleaned data to demonstrate the impact of outlier removal on the statistical findings.

Advanced Visualization: Generates plots that include the segmented time-series data along with annotations displaying the estimated change and p-value for each sensor, providing a rich, all-in-one analytical summary.

Key Libraries

Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn

Time-Series Modeling: statsmodels (for ARIMA), pmdarima (for auto_arima)

How to Run
Prerequisites

Python 3.x

A CSV file named sensor_readings_export.csv in the same directory.

Required libraries. Install them using pip:

code
Bash
download
content_copy
expand_less
pip install pandas matplotlib seaborn statsmodels pmdarima
Execution

The scripts are designed to be run sequentially to understand the analytical journey, but the most advanced and comprehensive script is cow_learn_again_3.py.

To see the full, final analysis:

code
Bash
download
content_copy
expand_less
python cow_learn_again_3.py

This will generate a series of .png files containing the final time-series plots, box plots, and correlation heatmaps, along with a detailed printout of the statistical intervention analysis results.

To walk through the process:

Start with cow_it.py and cow_it_1.py for initial EDA.

Run cow_learn_segments.py to see the data labeled by experiment.

Run the cow_learn_again.py series to see the statistical modeling evolve.

Core Concepts Demonstrated

End-to-End EDA: A comprehensive approach to understanding a new dataset.

Time-Series Preprocessing: Handling timestamps, indexing, cleaning, and segmentation.

Robust Outlier Detection: Using MAD to effectively handle noisy sensor data.

Intervention Analysis: A powerful statistical technique to measure the impact of an event on a time series.

Advanced Data Visualization: Creating clear, information-dense plots that combine raw data, segmented analysis, and statistical results.

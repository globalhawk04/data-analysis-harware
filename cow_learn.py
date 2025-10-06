import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

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

farm_start = "2025-04-17 12:16:56"
hay_start =  "2025-04-17 12:27:01"
cowpatty_start = "2025-04-17 12:42:05"
dirt_start = "2025-04-17 13:02:02"
grass_start ="2025-04-17 13:17:03"
farm_end = "2025-04-17 13:33:25"


# --- CORRECT METHOD: Use Index Slicing ---
# When your index is a DatetimeIndex, you can slice the DataFrame
# directly using strings or datetime objects that represent the index values.
# Slicing with datetime-like keys is inclusive of both start and end by default.

filtered_df = df[hay_start : farm_end]


print(f"Filtered DataFrame between {farm_start} and {hay_start} (inclusive using index slicing):")
print(filtered_df)

fig, axes = plt.subplots(nrows=len(filtered_df.columns), ncols=1, figsize=(12,15), sharex=True)

fig.suptitle('Sensor Readings Over Time',fontsize=16)

for i, col in enumerate(filtered_df.columns):
	axes[i].plot(filtered_df.index,filtered_df[col],label=col)
	axes[i].set_ylabel(col)
	axes[i].legend(loc='upper left')
	axes[i].grid(True)

plt.xlabel('Timestamp')
plt.tight_layout(rect=[0,0.03,1,0.98])
plt.savefig('cow_learn_1.png')



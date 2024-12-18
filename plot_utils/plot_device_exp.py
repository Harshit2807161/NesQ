'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Data for cd_N and cd_M (NesQ+ and Qroute results)
data_cd_N = {
    'radd_250': [2130, 1380, 1786],
    'z4_268': [1959, 1266, 1601],
    'rd73_252': [3256, 2200, 2786],
    'cycle10_2_110': [3891, 2627, 3244],
    'sqn_258': [6332, 4261, 5282]
}

data_cd_M = {
    'radd_250': [2145, 1554, 1757],
    'z4_268': [2423, 1368, 1645],
    'rd73_252': [3746, 2486, 2852],
    'cycle10_2_110': [4947, 2894, 3578],
    'sqn_258': [8097, 4941, 5417]
}

# Devices corresponding to rows (acorn, qx20, qx5)
devices = ['acorn', 'qx20', 'qx5']

# Convert data to DataFrames
df_cd_N = pd.DataFrame(data_cd_N, index=devices)
df_cd_M = pd.DataFrame(data_cd_M, index=devices)

# Compute the ratio of cd_N to cd_M, ensuring to handle division by zero or missing values
import numpy as np

# Ensure no division by zero issues (replace any potential NaN with zeros)
df_ratio = df_cd_N / df_cd_M
df_ratio = df_ratio.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
df_ratio = df_ratio.fillna(0)  # Replace NaN with 0 or another desired value

# Create a heatmap with larger annotation size and proper formatting
plt.figure(figsize=(10, 6))
sns.heatmap(df_ratio, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=0.5, vmin=0.8, vmax=1.1, annot_kws={"size": 16})

# Set the plot title and labels
plt.xlabel('Circuit Type', fontsize=14)
plt.ylabel('Device', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()
'''


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data for cd_N and cd_M (NesQ+ and Qroute results)
data_cd_N = np.array([
    [2130, 1959, 3256, 3891, 6332],  # acorn
    [1380, 1266, 2200, 2627, 4261],  # qx20
    [1786, 1601, 2786, 3244, 5282]   # qx5
])

data_cd_M = np.array([
    [2145, 2423, 3746, 4947, 8097],  # acorn
    [1554, 1368, 2486, 2894, 4941],  # qx20
    [1757, 1645, 2852, 3578, 5417]   # qx5
])

# Devices corresponding to rows
devices = ['acorn', 'qx20', 'qx5']

# Circuit types corresponding to columns
circuits = ['radd_250', 'z4_268', 'rd73_252', 'cycle10_2_110', 'sqn_258']

# Compute the ratio of cd_N to cd_M, handling division by zero or missing values
ratio = np.divide(data_cd_N, data_cd_M, where=data_cd_M != 0)  # Avoid division by zero

# Replace NaN or inf values that result from division
ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)

# Create the heatmap with larger annotation size and proper formatting
plt.figure(figsize=(10, 6))
sns.heatmap(ratio, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=0.5, vmin=0.8, vmax=1.1, annot_kws={"size": 16})

# Set the plot title and labels
plt.xticks(ticks=np.arange(len(circuits)) + 0.5, labels=circuits, rotation=45, ha='right', fontsize=12)
plt.yticks(ticks=np.arange(len(devices)) + 0.5, labels=devices, rotation=0, fontsize=12)

plt.xlabel('Circuit Type', fontsize=14)
plt.ylabel('Device', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# --- Data Embedded Directly in Script ---
# This data combines both 'Graph Generated' and 'Handcrafted' performance.
# Note: 'time' for Graph Generated Epoch 1 is None as its duration isn't directly logged.
data_for_plotting = [
    #{'epoch': 1, 'accuracy': 9.758333333333333, 'time': None, 'method': 'Graph Generated'},
    {'epoch': 2, 'accuracy': 10.979166666666666, 'time': 27.843334, 'method': 'Graph Generated'},
    {'epoch': 3, 'accuracy': 14.127083333333335, 'time': 27.937499, 'method': 'Graph Generated'},
    {'epoch': 4, 'accuracy': 13.758333333333333, 'time': 28.481045, 'method': 'Graph Generated'},
    {'epoch': 5, 'accuracy': 15.26875, 'time': 30.015635, 'method': 'Graph Generated'},
    {'epoch': 6, 'accuracy': 17.727083333333332, 'time': 29.752503, 'method': 'Graph Generated'},
    {'epoch': 7, 'accuracy': 24.25, 'time': 28.407963, 'method': 'Graph Generated'},
    {'epoch': 8, 'accuracy': 34.08125, 'time': 29.300717, 'method': 'Graph Generated'},
    {'epoch': 9, 'accuracy': 47.17708333333333, 'time': 30.797301, 'method': 'Graph Generated'},
    {'epoch': 10, 'accuracy': 70.65, 'time': 29.239682, 'method': 'Graph Generated'},
    #{'epoch': 1, 'accuracy': 12.86, 'time': 26.1374, 'method': 'Handcrafted'},
    {'epoch': 2, 'accuracy': 14.23, 'time': 26.2423, 'method': 'Handcrafted'},
    {'epoch': 3, 'accuracy': 16.66, 'time': 26.29, 'method': 'Handcrafted'},
    {'epoch': 4, 'accuracy': 21.09, 'time': 26.3576, 'method': 'Handcrafted'},
    {'epoch': 5, 'accuracy': 23.28, 'time': 26.3557, 'method': 'Handcrafted'},
    {'epoch': 6, 'accuracy': 25.36, 'time': 26.3685, 'method': 'Handcrafted'},
    {'epoch': 7, 'accuracy': 28.52, 'time': 26.377, 'method': 'Handcrafted'},
    {'epoch': 8, 'accuracy': 34.50, 'time': 26.3927, 'method': 'Handcrafted'},
    {'epoch': 9, 'accuracy': 45.06, 'time': 26.3907, 'method': 'Handcrafted'},
    {'epoch': 10, 'accuracy': 60.33, 'time': 26.4182, 'method': 'Handcrafted'},
]

# Create DataFrame from the embedded data
df_combined = pd.DataFrame(data_for_plotting)

# --- Plotting with Matplotlib ---

plt.style.use('seaborn-v0_8-darkgrid') # Optional: for a nicer plot style

# Create the Accuracy vs. Epoch plot
plt.figure(figsize=(10, 6))
for method in df_combined['method'].unique():
    subset = df_combined[df_combined['method'] == method]
    plt.plot(subset['epoch'], subset['accuracy'], marker='o', label=method)

plt.title('Accuracy vs. Epoch for Graph Generated and Handcrafted Methods')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(df_combined['epoch'].unique()) # Ensure integer ticks for epochs
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./AccuracyPlot.png")

# Create the Time vs. Epoch plot
plt.figure(figsize=(10, 6))
for method in df_combined['method'].unique():
    subset = df_combined[df_combined['method'] == method]
    # For 'Graph Generated' time plot, exclude rows where 'time' is NaN/None (Epoch 1)
    if method == 'Graph Generated':
        subset_plot = subset.dropna(subset=['time'])
        plt.plot(subset_plot['epoch'], subset_plot['time'], marker='o', label=f"{method}")
    else:
        plt.plot(subset['epoch'], subset['time'], marker='o', label=method)

plt.title('Time vs. Epoch for Graph Generated and Handcrafted Methods')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.xticks(df_combined['epoch'].unique()) # Ensure integer ticks for epochs
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./TimePlot.png")
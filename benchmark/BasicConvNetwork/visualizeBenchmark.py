import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'text.usetex':
    False,  # Set to True if LaTeX is installed and desired for text rendering
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'patch.linewidth': 1.2,
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

# Load and prepare data
# Create a dummy DataFrame if benchmarkData.csv is not available
try:
    df = pd.read_csv("benchmarkData.csv")
except FileNotFoundError:
    print(
        "benchmarkData.csv not found. Creating dummy data for demonstration.")
    data = {
        "whom": ["synapse-works", "synapse-works", "standalone", "standalone"],
        "model_type": ["conv", "linear", "conv", "linear"],
        "average_cpu_time_us": [15e6, 12e6, 18e6, 10e6],  # in microseconds
        "average_cuda_time": [80e6, 65e6, 90e6, 70e6],  # in microseconds
        "average_accuracy": [0.985, 0.972, 0.991, 0.968],
        "average_construction_time_s": [0.5, 0.3, 0.7, 0.4],
        "clicks": [5, 3, 7, 4],
        "loc": [150, 100, 200, 80]
    }
    df = pd.DataFrame(data)

# Data preparation
df["average_cpu_time_s"] = df["average_cpu_time_us"] / 1e6
df["average_cuda_time_s"] = df["average_cuda_time"] / 1e6
df["average_accuracy_percent"] = df["average_accuracy"] * 100
df["model_config"] = df["whom"] + "\n" + df["model_type"]

# Create a new combined effort metric
df['development_effort'] = df.apply(
    lambda row: row['clicks']
    if row['whom'] == 'synapse-works' else row['loc'],
    axis=1)
df['development_effort_label'] = df.apply(
    lambda row: 'Clicks'
    if row['whom'] == 'synapse-works' else 'LOC',
    axis=1)

# Sharp, publication-quality color palette
colors = {
    'synapse-works': '#1f77b4',  # Professional blue
    'standalone': '#ff7f0e',  # Orange
    'conv': '#2ca02c',  # Green
    'linear': '#d62728'  # Red
}


def create_clean_plot(title, figsize=(10, 6)):
    """Create a clean, publication-ready plot"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_title(title, fontweight='bold', pad=20, fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    return fig, ax


# ============================================================================
# CHART 1: PERFORMANCE METRICS (with Dual Y-Axes)
# ============================================================================
fig1, ax1 = create_clean_plot('Training Performance Comparison', (12, 6))

x = np.arange(len(df))
width = 0.25  # Changed from 0.15 to 0.25

# Plot CPU Time on the primary y-axis (ax1)
bars1 = ax1.bar(x - width / 2,
                df['average_cpu_time_s'],
                width,
                label='CPU Time',
                color=colors['synapse-works'],
                alpha=0.8,
                edgecolor='black',
                linewidth=1)
ax1.set_ylabel('CPU Time (seconds)',
               fontweight='bold',
               fontsize=12,
               color=colors['synapse-works'])
ax1.tick_params(axis='y', labelcolor=colors['synapse-works'])
ax1.set_ylim(bottom=0)  # Ensure y-axis starts from 0

# Create a second y-axis for CUDA Time
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width / 2,
                     df['average_cuda_time_s'],
                     width,
                     label='CUDA Time',
                     color=colors['standalone'],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1)
ax1_twin.set_ylabel('CUDA Time (seconds)',
                    fontweight='bold',
                    fontsize=12,
                    color=colors['standalone'])
ax1_twin.tick_params(axis='y', labelcolor=colors['standalone'])
ax1_twin.set_ylim(bottom=0)  # Ensure y-axis starts from 0

# Add value labels for CPU Time
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}s',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center',
                 va='bottom',
                 fontsize=9,
                 fontweight='bold',
                 color=colors['synapse-works'])

# Add value labels for CUDA Time
for bar in bars2:
    height = bar.get_height()
    ax1_twin.annotate(f'{height:.2f}s',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center',
                      va='bottom',
                      fontsize=9,
                      fontweight='bold',
                      color=colors['standalone'])

ax1.set_xlabel('Model Configuration', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(df['model_config'], rotation=0, ha='center')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1_twin.legend(lines + lines2,
                labels + labels2,
                loc='upper left',
                frameon=True,
                fancybox=True,
                shadow=True)

plt.tight_layout()
plt.savefig('1_performance_metrics.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.savefig('1_performance_metrics.pdf',
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.show()

# ============================================================================
# CHART 2: CONSTRUCTION EFFORT METRICS (Merged Clicks and LOC)
# ============================================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5),
                                  facecolor='white')  # Changed to 2 subplots
fig2.suptitle('Development & Construction Effort Analysis',
              fontweight='bold',
              fontsize=16,
              y=1.02)

# Construction Time (remains the same)
bars_const = ax2a.bar(df['model_config'],
                      df['average_construction_time_s'],
                      color=colors['conv'],
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=1)
ax2a.set_title('Model Construction Time', fontweight='bold', pad=15)
ax2a.set_ylabel('Time (seconds)', fontweight='bold')
ax2a.grid(True, alpha=0.3, axis='y')
ax2a.set_axisbelow(True)
ax2a.set_ylim(bottom=0)

# Combined Development Effort (Clicks for Synapse-works, LOC for Standalone)
bars_effort = ax2b.bar(df['model_config'],
                       df['development_effort'],
                       color='#9467bd',
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=1)  # Using a new color
ax2b.set_title('Development Effort', fontweight='bold', pad=15)
ax2b.set_ylabel('Metric Value', fontweight='bold')  # Generic label
ax2b.grid(True, alpha=0.3, axis='y')
ax2b.set_axisbelow(True)
ax2b.set_ylim(bottom=0)

# Add value labels for all construction metrics
for ax, bars, metric_col in [(ax2a, bars_const, None),
                             (ax2b, bars_effort, 'development_effort_label')]:
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label_text = f'{int(height)}' if height >= 1 else f'{height:.2f}'
        if metric_col:  # For the combined effort chart, add the specific label (Clicks/LOC)
            label_text += f' ({df[metric_col].iloc[i]})'
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold')

# Rotate x-axis labels for better readability
for ax in [ax2a, ax2b]:  # Only two subplots now
    ax.tick_params(axis='x', rotation=0, labelsize=10)

plt.tight_layout()
plt.savefig('2_construction_effort.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.savefig('2_construction_effort.pdf',
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.show()

# ============================================================================
# CHART 3: ACCURACY & QUALITY METRICS
# ============================================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
fig3.suptitle('Model Quality & Performance Trade-offs',
              fontweight='bold',
              fontsize=16,
              y=1.02)

# Accuracy Comparison
bars_acc = ax3a.bar(df['model_config'],
                    df['average_accuracy_percent'],
                    color='#8c564b',
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.15)
ax3a.set_title('Model Accuracy Comparison', fontweight='bold', pad=15)
ax3a.set_ylabel('Accuracy (%)', fontweight='bold')
ax3a.set_ylim(60, 100)
ax3a.grid(True, alpha=0.3, axis='y')
ax3a.set_axisbelow(True)

# Add accuracy value labels
for bar in bars_acc:
    height = bar.get_height()
    ax3a.annotate(f'{height:.2f}%',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),
                  textcoords="offset points",
                  ha='center',
                  va='bottom',
                  fontsize=9,
                  fontweight='bold')


from matplotlib.lines import Line2D

total_times = df['average_cpu_time_s'] + df['average_cuda_time_s']

# Define markers
def get_marker(model_type):
    if 'unet' in model_type.lower():
        return '^'
    elif 'conv' in model_type.lower():
        return 'o'
    else:
        return 's'

# Plot
for i, (_, row) in enumerate(df.iterrows()):
    marker = get_marker(row['model_type'])
    color = colors['synapse-works'] if 'synapse' in row['whom'] else colors['standalone']
    size = 150

    ax3b.scatter(
        total_times.iloc[i],
        row['average_accuracy_percent'],
        s=size,
        c=color,
        marker=marker,
        alpha=0.85,
        edgecolors='black',
        linewidth=0.2
    )

    # Optional: Shorten label or use index
    label = f"{i+1}"  # You can replace this with an abbreviation if preferred
    ax3b.annotate(
        label,
        xy=(total_times.iloc[i], row['average_accuracy_percent']),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=8,
        alpha=0.9,
        fontweight='bold'
    )

ax3b.set_title('Performance vs Accuracy Trade-off', fontweight='bold', pad=15)
ax3b.set_xlabel('Total Training Time (seconds)', fontweight='bold')
ax3b.set_ylabel('Accuracy (%)', fontweight='bold')
ax3b.grid(True, alpha=0.3)
ax3b.set_axisbelow(True)

# --- Updated Legend ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Conv Model',
           markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='Linear Model',
           markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='U-Net Model',
           markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['synapse-works'],
           label='Synapse-works', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['standalone'],
           label='Standalone', markeredgecolor='black', markersize=10),
]

ax3b.legend(
    handles=legend_elements,
    loc='lower left',
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize=10
)

plt.tight_layout()
plt.savefig('3_accuracy_quality.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('3_accuracy_quality.pdf', bbox_inches='tight', facecolor='white')
plt.show()
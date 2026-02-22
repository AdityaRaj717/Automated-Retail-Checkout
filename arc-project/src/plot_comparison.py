import matplotlib.pyplot as plt
import numpy as np

# Data from your output
models = ['MobileNetV2', 'RetailAttnNet (Ours)']
params = [2.23, 2.80]  # In Millions
sizes = [8.74, 10.75]  # In MB

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Parameters (Bar 1)
rects1 = ax1.bar(x - width/2, params, width, label='Parameters (Millions)', color='#3b82f6', alpha=0.8)
ax1.set_ylabel('Parameters (Millions)', fontsize=12)
ax1.set_title('Architecture Comparison: General vs. Custom', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_ylim(0, 3.5)

# Plot Size (Bar 2) on secondary Y-axis
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, sizes, width, label='Model Size (MB)', color='#ef4444', alpha=0.8)
ax2.set_ylabel('Size (MB)', fontsize=12)
ax2.set_ylim(0, 13)

# Add data labels
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

# Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.savefig("model_comparison.png")
print("Comparison plot saved as 'model_comparison.png'")
plt.show()

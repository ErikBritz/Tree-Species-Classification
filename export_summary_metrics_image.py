import re
import matplotlib.pyplot as plt

# Path to your test_report.txt
report_path = "training_results/test_report.txt"

# Read the report
with open(report_path, "r") as f:
    text = f.read()

# Extract accuracy, macro avg, weighted avg
accuracy_match = re.search(r"accuracy\s+([0-9.]+)", text)
macro_match = re.search(r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text)
weighted_match = re.search(
    r"weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text)

accuracy = float(accuracy_match.group(1)) if accuracy_match else None
macro_f1 = float(macro_match.group(3)) if macro_match else None
weighted_f1 = float(weighted_match.group(3)) if weighted_match else None

# Prepare data for vertical bar plot
labels = ["Accuracy", "Macro F1", "Weighted F1"]
values = [accuracy, macro_f1, weighted_f1]

fig, ax = plt.subplots(figsize=(4, 6))
bars = ax.barh(labels, values, color=["#4C72B0", "#55A868", "#C44E52"])

# Annotate bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f"{width:.3f}", va='center', fontsize=12)

ax.set_xlim(0, 1)
ax.set_xlabel("Score")
ax.set_title("Test Set Summary Metrics")
plt.tight_layout()
plt.savefig("training_results/summary_metrics.png")
plt.close()
print("Exported summary_metrics.png to training_results/")

import json
import pandas as pd

# Path to your metrics file
metrics_path = "training_results/test_metrics.json"

# Load metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Build a DataFrame for per-class metrics
# Ensure all keys are sorted the same way
class_names = list(metrics["per_class_f1"].keys())
df = pd.DataFrame({
    "F1": [metrics["per_class_f1"][k] for k in class_names],
    "Precision": [metrics["per_class_precision"][k] for k in class_names],
    "Recall": [metrics["per_class_recall"][k] for k in class_names],
    "Correct": [metrics["per_class_correct"][k] for k in class_names],
}, index=class_names)

# Print a pretty table
print(df.to_markdown(tablefmt="grid", floatfmt=".3f"))

# Optionally, save as CSV or Markdown
out_dir = "training_results/"
df.to_csv(out_dir + "per_class_metrics.csv")
df.to_markdown(out_dir + "per_class_metrics.md")

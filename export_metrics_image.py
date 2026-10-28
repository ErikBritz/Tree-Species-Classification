import json
import pandas as pd
import dataframe_image as dfi

# Path to your metrics file
metrics_path = "training_results/test_metrics.json"

# Load metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Build a DataFrame for per-class metrics
class_names = list(metrics["per_class_f1"].keys())
df = pd.DataFrame({
    "F1": [metrics["per_class_f1"][k] for k in class_names],
    "Precision": [metrics["per_class_precision"][k] for k in class_names],
    "Recall": [metrics["per_class_recall"][k] for k in class_names],
    "Correct": [metrics["per_class_correct"][k] for k in class_names],
}, index=class_names)

# Style the DataFrame for better appearance
styled_df = df.style.background_gradient(cmap="Blues").format("{:.3f}")

# Export as PNG image
out_path = "training_results/per_class_metrics.png"
dfi.export(styled_df, out_path)
print(f"Exported pretty table to {out_path}")

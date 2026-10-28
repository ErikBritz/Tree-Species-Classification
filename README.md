# CNN + Metadata for Tree-Species Classification

A PyTorch project for classifying tree species from multispectral satellite
patches and seasonal/topographic metadata. The image branch is a compact CNN;
an MLP encodes metadata, and the two feature vectors are fused for
classification.

## What this project demonstrates

- Multispectral patch loading and memory-mapped NumPy arrays
- Train-only per-band normalization and stratified train/validation/test splits
- Class-weighted loss and optional weighted sampling for imbalanced classes
- Optional spatial/noise augmentation
- Early stopping using validation macro-F1
- Per-class reports, metrics, and confusion-matrix plots

## Data

The training pipeline expects a prepared dataset directory containing:

```text
config.json
class_names.json
patches_f16.npy
meta_f16.npy
months_u1.npy
id_y_i2.npy
```

See `processed_dataset/config.json` and `processed_dataset/class_names.json`
in your local copy for the expected schema. Dataset arrays and source Parquet
files are intentionally excluded from Git: they are large, may have separate
terms of use, and should be obtained from their original source. Check the
dataset license and attribution requirements before using or redistributing it.
The `convert_test_parquet_to_numpy.py` utility converts a compatible Parquet
test set; it does not create the training dataset.

## Setup

Python 3.10 or newer is recommended. Create and activate a virtual environment,
then install the core dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

Run from the project root:

```powershell
python -m src.train --dataset-dir .\path\to\prepared_dataset --out-dir .\runs\baseline
```

For example, if you have supplied the prepared arrays locally:

```powershell
python -m src.train --dataset-dir .\processed_dataset --out-dir .\runs\baseline --use-sampler
```

Useful options include `--epochs`, `--batch-size`, `--patience`,
`--class-weight-power`, `--test-months`, and `--augment`. Run
`python -m src.train --help` for the full list. Training uses CUDA when
available and otherwise runs on CPU.

Each run writes a checkpoint, normalization statistics, training history,
test metrics/report, and a confusion-matrix plot under the output directory.
Generated runs and model checkpoints are ignored by Git.

## Utilities

- `convert_test_parquet_to_numpy.py` — convert compatible per-species Parquet
  files to the array format used by the pipeline.
- `validate_test_parquet.py` — inspect Parquet columns, sizes, and missing
  values without printing full image patches.
- `inspect_processed_dataset.py` — inspect array shapes and dtypes.
- `pretty_metrics_table.py`, `export_metrics_image.py`, and
  `export_summary_metrics_image.py` — export tables and metric plots.
- `export_torchviz_model_graph.py` and `src/visualize_pipeline.py` — optional
  architecture visualizations.

Optional visualization dependencies are listed in
`requirements-visualization.txt`. Rendering the TorchViz graph also requires
the Graphviz `dot` executable on `PATH`.

## Existing evaluation snapshot

The local `test_results/test_report.txt` records a macro-F1 of **0.1150** and
accuracy of **0.1784** over 16,583 examples. The original evaluation procedure
is not fully captured in the current scripts, so treat these as historical
results rather than a directly reproducible benchmark. The low scores also
highlight the difficulty of generalizing across imbalanced species classes.

## Repository contents

`src/` contains the reusable dataset, model, metrics, and training modules.
Large datasets, generated outputs, virtual environments, and unrelated course
documents are kept out of the versioned project.

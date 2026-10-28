import matplotlib.pyplot as plt
import numpy as np


def visualize_batch(batch, band_names=None, n_samples=4):
    """
    Visualize a batch from PatchMetaDataset.
    Shows the first n_samples patches and their meta features.
    Args:
        batch: a batch from DataLoader (with .patch, .meta, .y, .month)
        band_names: list of band/channel names (optional)
        n_samples: number of samples to visualize
    """
    patches = batch.patch.cpu().numpy()  # (B, C, H, W)
    metas = batch.meta.cpu().numpy()     # (B, M)
    ys = batch.y.cpu().numpy()           # (B,)
    months = batch.month.cpu().numpy()   # (B,)
    batch_size, channels, _, _ = patches.shape
    n_samples = min(n_samples, batch_size)

    _, axes = plt.subplots(n_samples, 2, figsize=(6, 3 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n_samples):
        # Show first 3 bands as RGB if possible, else first band
        if channels >= 3:
            img = np.stack([
                patches[i, 0],  # e.g. B3
                patches[i, 1],  # e.g. B4
                patches[i, 2],  # e.g. B5
            ], axis=-1)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        else:
            img = patches[i, 0]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Patch {i} | y={ys[i]} | month={months[i]}")
        axes[i, 0].axis('off')
        # Show meta features
        axes[i, 1].bar(np.arange(metas.shape[1]), metas[i])
        axes[i, 1].set_title(f"Meta features {i}")
        if band_names is not None:
            axes[i, 0].set_xlabel(f"Bands: {', '.join(band_names[:channels])}")
    plt.tight_layout()
    plt.show()

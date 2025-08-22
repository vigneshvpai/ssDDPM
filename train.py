import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data.DWIDataLoader import DWIDataLoader

# Initialize the DWIDataLoader and get a batch
dataloader_module = DWIDataLoader(batch_size=1)
dataloader_module.setup()
dataloader = dataloader_module.train_dataloader()

# Fetch one batch
batch = next(iter(dataloader))  # Shape: (1, slices*bvalues, height, width)
print(f"Original batch shape: {batch.shape}")

# Remove batch dimension
batch_np = batch.squeeze(0)  # (slices*bvalues, height, width)
num_images, height, width = batch_np.shape

# Plot the first 100 images
num_to_plot = min(100, num_images)
rows = 10
cols = 10
fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
for i in range(num_to_plot):
    ax = axes[i // cols, i % cols]
    ax.imshow(batch_np[i], cmap="gray")
    ax.axis("off")
    ax.set_title(f"Slice {i}")
# Hide any unused subplots
for i in range(num_to_plot, rows * cols):
    ax = axes[i // cols, i % cols]
    ax.axis("off")
plt.tight_layout()
plt.savefig("preview.png")

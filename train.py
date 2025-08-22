import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data.DWIDataLoader import DWIDataLoader

# Initialize the DWIDataLoader and get a batch
dataloader_module = DWIDataLoader(batch_size=1)
dataloader_module.setup()
dataloader = dataloader_module.train_dataloader()

# Fetch one batch
batch = next(iter(dataloader))  # Expected shape: (1, slices, bvalues, height, width)
print(f"Batch shape: {batch.shape}")

# Move batch to numpy and remove batch dimension
batch_np = batch.squeeze(0).numpy()  # (slices, bvalues, height, width)
num_slices, bvalues, height, width = batch_np.shape

# Select the middle-most b value
middle_b_idx = bvalues // 2

# Plot all slices for the middle-most b value
fig, axes = plt.subplots(num_slices, 1, figsize=(4, 4 * num_slices))
if num_slices == 1:
    axes = np.array([axes])

for i in range(num_slices):
    ax = axes[i]
    ax.imshow(batch_np[i, middle_b_idx, :, :], cmap="gray")
    ax.set_title(f"Slice {i}, b={middle_b_idx}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("batch_slices_middle_bvalue.png", dpi=150, bbox_inches="tight")
print(f"Saved batch_slices_middle_bvalue.png to root directory")
print(
    f"Batch numpy shape: {batch_np.shape}, Num slices: {num_slices}, Num bvalues: {bvalues}, Middle b index: {middle_b_idx}"
)

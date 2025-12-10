import numpy as np
import os

data_dir = "dataset/nonogram_10x10_poc/train"
inputs = np.load(os.path.join(data_dir, "train__inputs.npy"))
labels = np.load(os.path.join(data_dir, "train__labels.npy"))

print(f"Inputs shape: {inputs.shape}")
print(f"Labels shape: {labels.shape}")

# Reshape to (N, 200)
inputs = inputs.reshape(-1, 200)
labels = labels.reshape(-1, 200)

print(f"Sample 0 Input: {inputs[0]}")
print(f"Sample 0 Label: {labels[0]}")

# Check if inputs contain clues and grid
# Clues should be integers > 0? Or 0-15?
# Grid should be 0 or 1?
print(f"Input unique values: {np.unique(inputs)}")
print(f"Label unique values: {np.unique(labels)}")

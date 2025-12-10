
import numpy as np

path = "dataset/nonogram_15x15/train/train__labels.npy"
print(f"Loading {path}...")
targets = np.load(path)

# Filter out ignore_index (-100)
valid_mask = targets != -100
valid_targets = targets[valid_mask]
total_valid = valid_targets.size
zeros = (valid_targets == 0).sum()
ones = (valid_targets == 1).sum()

print(f"Total valid cells: {total_valid}")
print(f"Zeros: {zeros} ({zeros/total_valid:.4f})")
print(f"Ones: {ones} ({ones/total_valid:.4f})")

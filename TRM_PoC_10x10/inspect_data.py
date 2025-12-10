import numpy as np
import os

data_path = "dataset/nonogram_10x10_poc/train"

print(f"Loading data from {data_path}...")

inputs = np.load(os.path.join(data_path, "train__inputs.npy"))
labels = np.load(os.path.join(data_path, "train__labels.npy"))

print(f"Inputs shape: {inputs.shape}")
print(f"Labels shape: {labels.shape}")

# Reshape to (N, 200)
inputs = inputs.reshape(-1, 200)
labels = labels.reshape(-1, 200)

print(f"Reshaped Inputs: {inputs.shape}")
print(f"Reshaped Labels: {labels.shape}")

sample_idx = 0
sample_input = inputs[sample_idx]
sample_label = labels[sample_idx]

print("\n--- Sample 0 ---")
print("Input (First 100 - Clues):")
print(sample_input[:100])
print("Input (Last 100 - Grid):")
print(sample_input[100:])

print("\nLabel (First 100 - Ignored):")
print(sample_label[:100])
print("Label (Last 100 - Grid):")
print(sample_label[100:])

print("\nStats:")
print(f"Input Grid Min/Max: {sample_input[100:].min()}/{sample_input[100:].max()}")
print(f"Label Grid Min/Max: {sample_label[100:].min()}/{sample_label[100:].max()}")

# Check for 0s
print(f"\nNumber of 0s in Input Grid: {np.sum(sample_input[100:] == 0)}")
print(f"Number of 1s in Input Grid: {np.sum(sample_input[100:] == 1)}")

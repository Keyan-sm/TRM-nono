
import numpy as np
import os

input_dir = "dataset/nonogram_15x15"
# The build script loaded these:
# input_path = os.path.join(input_dir, f'x_{subset_name}_15x15_ok.npz')
# But those are in the PARENT directory usually?
# Let's check where the build script expected them.
# The user ran build_dataset with some args.
# Let's just check the 'train__inputs.npy' we already have.
# It is flattened.
# But we can infer from seq_len = 465 and grid = 225.
# 465 - 225 = 240.
# Let's verify if the first 240 are clues and last 225 are grid.
# We can load a small chunk of 'train__inputs.npy' and inspect values.
# Grid values are 0 or 1.
# Clue values are 0-15 (and maybe -100 for padding? No, build script said inputs are int64).
# build_dataset: "inputs = inputs.astype(np.int64)"
# "targets = targets.astype(np.int64)"
# "full_inputs = np.concatenate([inputs, targets], axis=1)"
# So yes, it is [Clues, Grid].
# We just need to confirm Clues length is 240.

path = "dataset/nonogram_15x15/train/train__inputs.npy"
data = np.load(path, mmap_mode='r')
# Reshape first 465 items
sample = data[:465]
print(f"Sample shape: {sample.shape}")
print(f"First 240 (Clues?): {sample[:240]}")
print(f"Last 225 (Grid?): {sample[240:]}")
print(f"Grid unique values: {np.unique(sample[240:])}")
print(f"Clues unique values: {np.unique(sample[:240])}")

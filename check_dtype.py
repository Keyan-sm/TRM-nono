import numpy as np
import os

# Correct path relative to TRMProj root
data_path = "TRM_PoC_15x15/dataset/nonogram_15x15/train/train__inputs.npy"
try:
    data = np.load(data_path, mmap_mode='r')
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Itemsize: {data.dtype.itemsize}")
except Exception as e:
    print(f"Error: {e}")

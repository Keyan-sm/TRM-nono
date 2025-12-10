
import numpy as np
import os

path = "dataset/nonogram_15x15/train/train__inputs.npy"
try:
    data = np.load(path)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
except Exception as e:
    print(f"Error loading: {e}")

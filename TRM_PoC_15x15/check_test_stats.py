import numpy as np
import os

data_path = "dataset/nonogram_15x15/test/test__inputs.npy"
print(f"Checking data stats for {data_path}...")

try:
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        exit(1)

    data = np.load(data_path)
    print(f"Shape: {data.shape}")
    print(f"Min: {data.min()}")
    print(f"Max: {data.max()}")
    
    # Check for NaN/Inf
    if np.isnan(data).any():
        print("WARNING: Data contains NaNs!")
    if np.isinf(data).any():
        print("WARNING: Data contains Infs!")
        
    # Check unique values to see vocab usage
    unique_vals = np.unique(data)
    print(f"Unique values count: {len(unique_vals)}")
    print(f"First 20 unique values: {unique_vals[:20]}")
    print(f"Last 20 unique values: {unique_vals[-20:]}")
    
    # Check against vocab size 32
    if data.max() >= 32:
        print("CRITICAL: Data contains values >= 32!")
        
except Exception as e:
    print(f"Error: {e}")

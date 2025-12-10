import numpy as np
import os

x_path = '/Users/keyanmikaili/Downloads/TRMProj/TRM Context/NonoDataset-main/15x15/x_train_15x15_ok.npz'
y_path = '/Users/keyanmikaili/Downloads/TRMProj/TRM Context/NonoDataset-main/15x15/y_train_15x15_ok.npz'

def inspect(path, name):
    print(f"Inspecting {name}: {path}")
    try:
        with np.load(path) as data:
            print("Keys:", data.files)
            for key in data.files:
                print(f"  {key}: {data[key].shape}")
                sample = data[key][0]
                print(f"  Sample 0 (first 50): {sample.flatten()[:50]}")
    except Exception as e:
        print(f"Error: {e}")

inspect(x_path, "X")
inspect(y_path, "Y")

import numpy as np
import os

base_path_10 = 'TRM Context/NonoDataset-main/10x10'
icons_path = os.path.join(base_path_10, 'y_train_dataset.npz')

def inspect(path):
    print(f"Inspecting {path}")
    try:
        with np.load(path) as data:
            print("Keys:", data.files)
            for key in data.files:
                print(f"  {key}: {data[key].shape}")
                sample = data[key][0]
                print(f"  Sample 0:\n{sample}")
    except Exception as e:
        print(f"Error: {e}")

inspect(icons_path)

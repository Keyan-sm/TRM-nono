import numpy as np
import os
import json

data_dir = "dataset/nonogram_10x10_poc/test"
limit = 100
seq_len = 200

print(f"Truncating test dataset in {data_dir} to {limit} samples...")

# Truncate .npy files
for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        path = os.path.join(data_dir, filename)
        data = np.load(path)
        
        if "inputs" in filename or "labels" in filename:
            # Flattened: limit * seq_len
            new_len = limit * seq_len
            if data.shape[0] > new_len:
                data = data[:new_len]
                np.save(path, data)
        elif "puzzle_indices" in filename:
            # Indices: limit + 1
            # Recalculate indices
            # [0, 200, 400, ...]
            new_indices = np.arange(0, limit * seq_len + 1, seq_len, dtype=np.int64)
            np.save(path, new_indices)
        elif "group_indices" in filename:
            # [0, limit]
            new_indices = np.array([0, limit], dtype=np.int64)
            np.save(path, new_indices)
        elif "puzzle_identifiers" in filename:
            # limit
            if data.shape[0] > limit:
                data = data[:limit]
                np.save(path, data)

# Update dataset.json
json_path = os.path.join(data_dir, "dataset.json")
with open(json_path, "r") as f:
    metadata = json.load(f)

metadata["total_puzzles"] = limit
metadata["mean_puzzle_examples"] = float(limit)

with open(json_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("Done.")

import os
import json
import argparse
import numpy as np
from pathlib import Path

def build_dataset(input_dir, output_dir, subset_name='train', limit=None):
    print(f"Building {subset_name} dataset from {input_dir}...")
    
    # Define paths
    if subset_name == 'train':
        input_path = os.path.join(input_dir, f'x_{subset_name}_15x15_ok.npz')
        target_path = os.path.join(input_dir, f'y_{subset_name}_15x15_ok.npz')
    else:
        input_path = os.path.join(input_dir, f'x_{subset_name}_dataset.npz')
        target_path = os.path.join(input_dir, f'y_{subset_name}_dataset.npz')
    
    # Load data
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return
    
    print(f"Loading {input_path}...")
    with np.load(input_path) as data:
        inputs = data['arr_0']
        
    print(f"Loading {target_path}...")
    with np.load(target_path) as data:
        targets = data['arr_0']
        
    # Cast to int64 to avoid overflow when using -100
    inputs = inputs.astype(np.int64)
    targets = targets.astype(np.int64)
        
    # Validation
    if inputs.shape[0] != targets.shape[0]:
        print(f"Error: Mismatch in number of samples. Inputs: {inputs.shape[0]}, Targets: {targets.shape[0]}")
        return
        
    total_samples = inputs.shape[0]
    if limit:
        total_samples = min(total_samples, limit)
        inputs = inputs[:total_samples]
        targets = targets[:total_samples]
        
    print(f"Processing {total_samples} samples...")
    
    # Flatten inputs and targets if they aren't already (they should be (N, 100))
    # The dataset analysis says they are flattened arrays.
    # But let's check shape.
    # If they are (N, 100), we flatten to (N*100,) for the dataset loader?
    # puzzle_dataset.py:
    # batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))
    # It seems to sample INDICES from the puzzle range.
    # And `_collate_batch` takes `dataset["inputs"][batch_indices]`.
    # So `inputs` should be a 1D array of ALL tokens concatenated?
    # Or `inputs` is a list of arrays?
    # "inputs": np.load(..., mmap_mode="r")
    # Usually np.load returns an array.
    # If it's a 1D array, then `puzzle_indices` marks the start/end.
    
    # Concatenate inputs (clues) and targets (grid) to form full sequences
    # inputs: (N, 100), targets: (N, 100) -> full: (N, 200)
    
    # For inputs: [clues, grid] (Teacher forcing / Auto-regressive)
    # Actually, usually we feed [clues, grid] as input, and predict [clues, grid] (shifted) or just masked.
    # TRM likely expects standard causal masking.
    full_inputs = np.concatenate([inputs, targets], axis=1)
    
    # For labels: [-100 (ignore clues), grid]
    ignore_clues = np.full_like(inputs, -100)
    full_labels = np.concatenate([ignore_clues, targets], axis=1)
    
    print(f"Targets stats - Min: {targets.min()}, Max: {targets.max()}, Mean: {targets.mean()}")
    print(f"Full Labels stats - Min: {full_labels.min()}, Max: {full_labels.max()}, Mean: {full_labels.mean()}")
    
    seq_len = full_inputs.shape[1] # Should be 200 for 10x10
    
    flat_inputs = full_inputs.flatten()
    flat_targets = full_labels.flatten()
    
    # Create indices
    # puzzle_indices: [0, 100, 200, ..., N*100]
    puzzle_indices = np.arange(0, total_samples * seq_len + 1, seq_len, dtype=np.int64)
    
    # group_indices: [0, total_samples] (One big group)
    group_indices = np.array([0, total_samples], dtype=np.int64)
    
    # puzzle_identifiers: [0, 0, ..., 0] (One identifier for now)
    # This needs to be one per puzzle? Or one per token?
    # puzzle_dataset.py:
    # batch["puzzle_identifiers"] = dataset["puzzle_identifiers"][puzzle_indices]
    # It seems to be one per puzzle.
    puzzle_identifiers = np.zeros(total_samples, dtype=np.int32)
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, subset_name), exist_ok=True)
    
    # Save .npy files
    print("Saving .npy files...")
    np.save(os.path.join(output_dir, subset_name, f"{subset_name}__inputs.npy"), flat_inputs)
    np.save(os.path.join(output_dir, subset_name, f"{subset_name}__labels.npy"), flat_targets)
    np.save(os.path.join(output_dir, subset_name, f"{subset_name}__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(output_dir, subset_name, f"{subset_name}__group_indices.npy"), group_indices)
    np.save(os.path.join(output_dir, subset_name, f"{subset_name}__puzzle_identifiers.npy"), puzzle_identifiers)
    
    # Create metadata
    metadata = {
        "pad_id": 0,
        "ignore_label_id": -100,
        "blank_identifier_id": 0,
        "vocab_size": 32, # Sufficient for clues (0-15) and grid (0-1)
        "seq_len": int(seq_len),
        "num_puzzle_identifiers": 1,
        "total_groups": 1,
        "mean_puzzle_examples": float(total_samples), # Average examples per group (since 1 group, it's total_samples)
        "total_puzzles": int(total_samples),
        "sets": [subset_name]
    }
    
    with open(os.path.join(output_dir, subset_name, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Done. Saved to {output_dir}/{subset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to NonoDataset directory (e.g., .../10x10)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for TRM dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    
    args = parser.parse_args()
    
    # Build train
    build_dataset(args.input_dir, args.output_dir, "train", args.limit)
    
    # Build test
    build_dataset(args.input_dir, args.output_dir, "test", args.limit)

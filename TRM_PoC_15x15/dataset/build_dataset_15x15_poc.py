#!/usr/bin/env python3
"""
Build a 10% subset of the 15x15 nonogram dataset for proof of concept training.
This creates a smaller dataset (~66k samples) from the full 661k training set.
"""

import os
import json
import numpy as np
from pathlib import Path

def create_poc_subset(input_dir, output_dir, subset_ratio=0.1):
    """
    Create a PoC subset by taking the first subset_ratio of samples.
    
    Args:
        input_dir: Path to full dataset (e.g., 'dataset/nonogram_15x15/train')
        output_dir: Path to output PoC dataset (e.g., 'dataset/nonogram_15x15_poc/train')
        subset_ratio: Ratio of samples to keep (default 0.1 for 10%)
    """
    print(f"Creating PoC dataset with {subset_ratio*100}% of samples...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Load metadata
    metadata_path = os.path.join(input_dir, 'dataset.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nOriginal dataset:")
    print(f"  Total puzzles: {metadata['total_puzzles']}")
    print(f"  Sequence length: {metadata['seq_len']}")
    
    # Calculate subset size
    original_total = metadata['total_puzzles']
    subset_total = int(original_total * subset_ratio)
    
    print(f"\nSubset dataset:")
    print(f"  Total puzzles: {subset_total} ({subset_ratio*100}%)")
    
    # Load and subset data files using memory mapping for efficiency
    print(f"\nLoading data files...")
    
    inputs = np.load(os.path.join(input_dir, 'train__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(input_dir, 'train__labels.npy'), mmap_mode='r')
    puzzle_indices = np.load(os.path.join(input_dir, 'train__puzzle_indices.npy'))
    group_indices = np.load(os.path.join(input_dir, 'train__group_indices.npy'))
    puzzle_identifiers = np.load(os.path.join(input_dir, 'train__puzzle_identifiers.npy'))
    
    seq_len = metadata['seq_len']
    subset_end_idx = subset_total * seq_len
    
    print(f"  Original inputs shape: {inputs.shape}")
    print(f"  Subset end index: {subset_end_idx}")
    
    # Subset the data
    subset_inputs = np.array(inputs[:subset_end_idx])
    subset_labels = np.array(labels[:subset_end_idx])
    subset_puzzle_indices = puzzle_indices[:subset_total + 1]  # +1 to include end boundary
    subset_group_indices = np.array([0, subset_total], dtype=np.int64)  # One group
    subset_puzzle_identifiers = puzzle_identifiers[:subset_total]
    
    print(f"\nSubset shapes:")
    print(f"  Inputs: {subset_inputs.shape}")
    print(f"  Labels: {subset_labels.shape}")
    print(f"  Puzzle indices: {subset_puzzle_indices.shape}")
    print(f"  Group indices: {subset_group_indices.shape}")
    print(f"  Puzzle identifiers: {subset_puzzle_identifiers.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save subset data
    print(f"\nSaving subset data to {output_dir}...")
    np.save(os.path.join(output_dir, 'train__inputs.npy'), subset_inputs)
    np.save(os.path.join(output_dir, 'train__labels.npy'), subset_labels)
    np.save(os.path.join(output_dir, 'train__puzzle_indices.npy'), subset_puzzle_indices)
    np.save(os.path.join(output_dir, 'train__group_indices.npy'), subset_group_indices)
    np.save(os.path.join(output_dir, 'train__puzzle_identifiers.npy'), subset_puzzle_identifiers)
    
    # Update and save metadata
    subset_metadata = metadata.copy()
    subset_metadata['total_puzzles'] = subset_total
    subset_metadata['mean_puzzle_examples'] = float(subset_total)
    
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(subset_metadata, f, indent=4)
    
    print(f"\n✓ PoC dataset created successfully!")
    print(f"  Location: {output_dir}")
    print(f"  Total samples: {subset_total}")
    print(f"  Total size: {subset_inputs.nbytes + subset_labels.nbytes:,} bytes")
    
    return subset_metadata

if __name__ == '__main__':
    # Default paths
    input_dir = 'dataset/nonogram_15x15/train'
    output_dir = 'dataset/nonogram_15x15_poc/train'
    
    # Create subset
    create_poc_subset(input_dir, output_dir, subset_ratio=0.1)

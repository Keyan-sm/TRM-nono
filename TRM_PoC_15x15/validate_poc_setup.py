#!/usr/bin/env python3
"""
Validation script to check that the PoC dataset and config are set up correctly
before running full training.
"""

import sys
import json
import numpy as np
from pathlib import Path

def validate_dataset():
    """Validate the PoC dataset structure and metadata."""
    print("=" * 60)
    print("VALIDATING PoC DATASET")
    print("=" * 60)
    
    dataset_dir = Path('dataset/nonogram_15x15_poc/train')
    
    # Check directory exists
    if not dataset_dir.exists():
        print(f"❌ ERROR: Dataset directory not found: {dataset_dir}")
        return False
    
    # Load metadata
    metadata_path = dataset_dir / 'dataset.json'
    if not metadata_path.exists():
        print(f"❌ ERROR: Metadata file not found: {metadata_path}")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n✓ Metadata loaded:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Check expected sample count
    total_puzzles = metadata['total_puzzles']
    expected_min = 60000
    expected_max = 70000
    
    if not (expected_min < total_puzzles < expected_max):
        print(f"❌ ERROR: Unexpected sample count: {total_puzzles}")
        print(f"   Expected between {expected_min} and {expected_max}")
        return False
    
    print(f"\n✓ Sample count in expected range")
    
    # Validate data files
    required_files = [
        'train__inputs.npy',
        'train__labels.npy',
        'train__puzzle_indices.npy',
        'train__group_indices.npy',
        'train__puzzle_identifiers.npy'
    ]
    
    seq_len = metadata['seq_len']
    expected_total_tokens = total_puzzles * seq_len
    
    print(f"\n✓ Validating data files:")
    for filename in required_files:
        filepath = dataset_dir / filename
        if not filepath.exists():
            print(f"  ❌ Missing: {filename}")
            return False
        
        data = np.load(filepath, mmap_mode='r')
        print(f"  ✓ {filename}: shape={data.shape}, dtype={data.dtype}")
        
        # Validate shapes
        if 'inputs' in filename or 'labels' in filename:
            if data.shape[0] != expected_total_tokens:
                print(f"    ❌ ERROR: Expected {expected_total_tokens} tokens, got {data.shape[0]}")
                return False
        elif 'puzzle_indices' in filename:
            if data.shape[0] != total_puzzles + 1:
                print(f"    ❌ ERROR: Expected {total_puzzles + 1} indices, got {data.shape[0]}")
                return False
        elif 'puzzle_identifiers' in filename:
            if data.shape[0] != total_puzzles:
                print(f"    ❌ ERROR: Expected {total_puzzles} identifiers, got {data.shape[0]}")
                return False
    
    print(f"\n{'='*60}")
    print("✓ DATASET VALIDATION PASSED")
    print(f"{'='*60}\n")
    return True

def validate_config():
    """Validate the PoC configuration file."""
    print("=" * 60)
    print("VALIDATING PoC CONFIG")
    print("=" * 60)
    
    config_path = Path('config/nonogram_15x15_poc.yaml')
    
    if not config_path.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        return False
    
    print(f"\n✓ Config file exists: {config_path}")
    
    # Read and display key settings
    with open(config_path, 'r') as f:
        content = f.read()
    
    key_settings = {
        'data_paths': "['dataset/nonogram_15x15_poc']",
        'global_batch_size': '64',
        'epochs': '10',
        'eval_interval': '1',
        'hidden_size': '512',
        'num_heads': '8',
        'H_cycles': '3',
        'L_cycles': '4',
        'L_layers': '2',
        'forward_dtype': '"float32"',
        'loss_type': 'softmax_cross_entropy'
    }
    
    print(f"\n✓ Checking key settings:")
    all_found = True
    for key, expected in key_settings.items():
        if expected in content:
            print(f"  ✓ {key}: {expected}")
        else:
            print(f"  ❌ {key}: not found or incorrect")
            all_found = False
    
    if not all_found:
        print(f"\n❌ ERROR: Some key settings missing or incorrect")
        return False
    
    print(f"\n{'='*60}")
    print("✓ CONFIG VALIDATION PASSED")
    print(f"{'='*60}\n")
    return True

def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("TRM 15x15 PoC SETUP VALIDATION")
    print("=" * 60 + "\n")
    
    dataset_ok = validate_dataset()
    config_ok = validate_config()
    
    if dataset_ok and config_ok:
        print("\n" + "=" * 60)
        print("✅ ALL VALIDATIONS PASSED - READY TO TRAIN!")
        print("=" * 60)
        print("\nTo start training, run:")
        print("  python3 pretrain.py --config-name nonogram_15x15_poc")
        print()
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED - PLEASE FIX ERRORS ABOVE")
        print("=" * 60 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())

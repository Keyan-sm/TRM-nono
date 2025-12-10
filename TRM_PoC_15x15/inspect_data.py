
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.simple_nonogram_dataset import SimpleNonogramDataset
import numpy as np
import sys
import os

# Mock hydra main to load config
@hydra.main(config_path="config", config_name="nonogram_15x15", version_base=None)
def inspect_data(config: DictConfig):
    print(f"Loading dataset for project: {config.project_name}")
    
    # Load dataset logic from pretrain.py
    try:
        if "nonogram" in config.data_paths[0]:
            print(f"Using SimpleNonogramDataset with path: {config.data_paths[0]}")
            dataset = SimpleNonogramDataset(config.data_paths[0], split="train")
            # metadata = dataset.puzzle_dataset_metadata
        else:
            print("Using PuzzleDataset")
            dataset = PuzzleDataset(PuzzleDatasetConfig(
                seed=config.seed,
                dataset_paths=config.data_paths,
                rank=0,
                num_replicas=1,
            ), split="train")
            
        dataloader = DataLoader(
            dataset,
            batch_size=config.global_batch_size, 
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Dataloader len: {len(dataloader)}")
    
    # Inspect first few batches
    for i, batch in enumerate(dataloader):
        if i >= 20: break
        
        print(f"\n--- Batch {i} ---")
        inputs = batch["inputs"]
        puzzle_ids = batch["puzzle_identifiers"]
        
        print(f"Inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
        print(f"Puzzle IDs shape: {puzzle_ids.shape}, dtype: {puzzle_ids.dtype}")
        
        # Check for NaNs/Infs
        if torch.isnan(inputs).any():
            print("WARNING: NaNs found in inputs!")
        if torch.isinf(inputs).any():
            print("WARNING: Infs found in inputs!")
            
        # Check value ranges
        print(f"Inputs Min: {inputs.min()}, Max: {inputs.max()}")
        print(f"Puzzle IDs Min: {puzzle_ids.min()}, Max: {puzzle_ids.max()}")
        
        # Check vocab size compliance
        vocab_size = 32 # Hardcoded as config access is tricky
        if inputs.max() >= vocab_size:
            print(f"CRITICAL: Input value {inputs.max()} exceeds vocab size {vocab_size}!")
        
        # Check for all zeros (might indicate empty data)
        if (inputs == 0).all():
            print("WARNING: Inputs are all zeros!")

    print("\nData inspection complete.")

if __name__ == "__main__":
    inspect_data()

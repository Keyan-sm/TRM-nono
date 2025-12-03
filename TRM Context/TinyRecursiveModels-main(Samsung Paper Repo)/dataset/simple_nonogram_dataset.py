import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class SimpleNonogramDataset(Dataset):
    def __init__(self, data_path, split="train"):
        self.data_path = data_path
        self.split = split
        
        # Load metadata
        print(f"Loading dataset from: {os.path.join(data_path, split)}")
        with open(os.path.join(data_path, split, "dataset.json"), "r") as f:
            self.metadata = json.load(f)
        print(f"Loaded metadata: {self.metadata}")
            
        self.seq_len = self.metadata["seq_len"]
        self.vocab_size = self.metadata["vocab_size"]
        self.num_puzzle_identifiers = self.metadata["num_puzzle_identifiers"]
        
        # Load data
        self.inputs = np.load(os.path.join(data_path, split, f"{split}__inputs.npy"))
        self.labels = np.load(os.path.join(data_path, split, f"{split}__labels.npy"))
        
        # Reshape to (N, SeqLen)
        # The build script saved them as flattened, but we know they are structured
        num_samples = self.inputs.shape[0] // self.seq_len
        self.inputs = self.inputs.reshape(num_samples, self.seq_len)
        self.labels = self.labels.reshape(num_samples, self.seq_len)
        
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return {
            "inputs": torch.from_numpy(self.inputs[idx]).long(),
            "labels": torch.from_numpy(self.labels[idx]).long(),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long) # Dummy identifier
        }

    @property
    def puzzle_dataset_metadata(self):
        # Return an object compatible with PuzzleDatasetMetadata
        from dataset.common import PuzzleDatasetMetadata
        return PuzzleDatasetMetadata(**self.metadata)

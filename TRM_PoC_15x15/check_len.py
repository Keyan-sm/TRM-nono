import torch
from torch.utils.data import DataLoader
from dataset.simple_nonogram_dataset import SimpleNonogramDataset
import os

data_path = "dataset/nonogram_15x15"
split = "train"
batch_size = 64

print(f"Checking dataset at {data_path}...")
try:
    dataset = SimpleNonogramDataset(data_path, split=split)
    print(f"Dataset length: {len(dataset)}")
    print(f"Seq len: {dataset.seq_len}")
    print(f"Inputs shape: {dataset.inputs.shape}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    print(f"Dataloader length: {len(dataloader)}")
except Exception as e:
    print(f"Error: {e}")

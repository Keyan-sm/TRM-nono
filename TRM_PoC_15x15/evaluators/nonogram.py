from typing import Dict, Sequence, Optional
import os
import json

import torch
import numpy as np
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata

class NonogramEvaluator:
    required_outputs = {"inputs", "labels", "preds"}
    
    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata):
        super().__init__()
        self.ignore_label_id = eval_metadata.ignore_label_id
        
    def begin_eval(self):
        self.total_correct = 0
        self.total_samples = 0
        self.total_cells = 0
        self.correct_cells = 0
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU
        labels = batch["labels"].cpu().numpy()
        predictions = preds["preds"].cpu().numpy()
        
        # Calculate accuracy
        # Mask out ignored labels
        mask = labels != self.ignore_label_id
        
        # Cell-wise accuracy
        self.correct_cells += np.sum((predictions == labels) & mask)
        self.total_cells += np.sum(mask)
        
        # Sample-wise accuracy (all cells in a sample must be correct)
        # We need to check per row
        for i in range(labels.shape[0]):
            sample_mask = mask[i]
            if np.sum(sample_mask) > 0: # Only count if there are valid labels
                if np.all(predictions[i][sample_mask] == labels[i][sample_mask]):
                    self.total_correct += 1
                self.total_samples += 1

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        # Gather stats to rank 0
        stats = torch.tensor([self.total_correct, self.total_samples, self.correct_cells, self.total_cells], dtype=torch.float64, device="cpu")
        
        if world_size > 1:
            # Move to device for reduction
            stats = stats.to(dist.get_backend(group) if group else "cuda") # Or mps/cpu
            dist.reduce(stats, dst=0)
            stats = stats.cpu()
            
        if rank == 0:
            total_correct = stats[0].item()
            total_samples = stats[1].item()
            correct_cells = stats[2].item()
            total_cells = stats[3].item()
            
            if total_samples == 0:
                return {}
                
            return {
                "test/accuracy": total_correct / total_samples,
                "test/cell_accuracy": correct_cells / total_cells
            }
        return None

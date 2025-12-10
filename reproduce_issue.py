import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict

# Mock Metadata
@dataclass
class PuzzleDatasetMetadata:
    ignore_label_id: int = -100

# Mock NonogramEvaluator to avoid importing the whole project structure if possible, 
# but better to import the actual class to be sure.
# However, importing might require setting up paths. 
# Let's try to import the actual class by adding the directory to sys.path.

import sys
import os

# Add TRM_PoC_15x15 to sys.path
sys.path.append(os.path.abspath("TRM_PoC_15x15"))

from evaluators.nonogram import NonogramEvaluator

def test_update_batch():
    print("Testing update_batch with mock data...")
    
    # Setup
    metadata = PuzzleDatasetMetadata(ignore_label_id=-100)
    evaluator = NonogramEvaluator(data_path=".", eval_metadata=metadata)
    evaluator.begin_eval()
    
    batch_size = 64
    seq_len = 465
    vocab_size = 32
    
    # Mock Batch
    # labels: (Batch, SeqLen)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Mock Preds
    # ACTLossHead returns preds as indices: (Batch, SeqLen)
    # This is what the model actually returns
    preds_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    batch = {"labels": labels}
    preds = {"preds": preds_indices}
    
    print(f"Mock Data Shapes:")
    print(f"  labels: {labels.shape}")
    print(f"  preds['preds']: {preds['preds'].shape}")
    
    try:
        evaluator.update_batch(batch, preds)
        print("SUCCESS: update_batch completed without error.")
    except ValueError as e:
        print(f"FAILURE: Caught expected ValueError: {e}")
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_update_batch()

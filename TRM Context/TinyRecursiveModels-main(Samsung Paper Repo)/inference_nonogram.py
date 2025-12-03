import torch
import hydra
import os
import numpy as np
from omegaconf import OmegaConf
from dataset.simple_nonogram_dataset import SimpleNonogramDataset
from pretrain import create_model
from models.layers import CastedLinear

@hydra.main(config_path="config", config_name="nonogram_10x10", version_base="1.2")
def main(config):
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load Dataset
    dataset = SimpleNonogramDataset(config.data_paths[0], split="train") # Use train for now as we don't have test split yet
    metadata = dataset.puzzle_dataset_metadata

    # Create Model (Manual adaptation for DictConfig)
    from utils.functions import load_model_class
    from models.losses import ACTLossHead
    
    # Convert DictConfig to dict
    arch_cfg = OmegaConf.to_container(config.arch, resolve=True)
    # Remove keys that might be in arch but not needed or handled separately if any
    # But here we just need to merge it with dataset params
    
    model_cfg = dict(
        **arch_cfg,
        batch_size=1, # Inference batch size
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
    )
    
    # Remove 'loss' from model_cfg as it's for the wrapper
    loss_cfg = model_cfg.pop("loss", None)
    
    # Instantiate model
    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    
    # Clean model_cfg
    model_cfg.pop("model_name", None)
    
    # Pass dict directly as expected by __init__
    model = model_cls(model_cfg)
    
    # Wrap with Loss Head
    if loss_cfg:
        # ACTLossHead only accepts loss_type
        # If loss_cfg has 'name' or other keys, remove them or extract loss_type
        loss_type = loss_cfg.get("loss_type")
        if loss_type:
            model = ACTLossHead(model, loss_type=loss_type)
        else:
            print("Warning: No loss_type found in loss_cfg, skipping LossHead wrapper (inference might fail if model output format differs)")
    
    model.to(device)
    model.eval()

    # Load Checkpoint
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "Nonogram_10x10-ACT-torch", "TinyRecursiveReasoningModel_ACTV1 khaki-mammoth", "step_124146")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found! Running with random weights.")

    # Get a sample
    sample = dataset[0]
    inputs = sample["inputs"].unsqueeze(0).to(device) # (1, SeqLen)
    labels = sample["labels"].unsqueeze(0).to(device) # (1, SeqLen)
    puzzle_identifiers = sample["puzzle_identifiers"].unsqueeze(0).to(device)

    # Run Inference
    with torch.no_grad():
        # Forward pass
        # The model expects a batch dict
        batch = {
            "inputs": inputs,
            "labels": labels,
            "puzzle_identifiers": puzzle_identifiers
        }
        
        # Initial carry
        carry = model.initial_carry(batch)
        
        # Move carry to device (workaround for trm.py bug)
        carry.steps = carry.steps.to(device)
        carry.halted = carry.halted.to(device)
        # inner_carry might need moving too. 
        # inner_carry is TinyRecursiveReasoningModel_ACTV1_InnerCarry
        # Let's assume it has fields z_H, etc.
        # We can try to move them if we know the structure or if it has a .to() method (unlikely for simple dataclass)
        # Let's inspect inner_carry structure from trm.py view earlier or just try to move known fields.
        # From previous error: z_H is on CPU.
        if hasattr(carry.inner_carry, "z_H"):
             carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        if hasattr(carry.inner_carry, "z_L"):
             carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        # There might be other fields.
        # Let's check trm.py for TinyRecursiveReasoningModel_ACTV1_InnerCarry definition.
        # But for now, let's try moving z_H.
        
        # Also current_data should be on device because empty_like uses input device.
        
        # Debug devices
        print(f"Model device: {next(model.parameters()).device}")
        print(f"H_init device: {model.model.inner.H_init.device}")
        print(f"carry.z_H device: {carry.inner_carry.z_H.device}")
        print(f"carry.halted device: {carry.halted.device}")
        
        # We need to manually construct the args for model forward if we don't use the loss wrapper's forward
        # But wait, create_model returns the ACTLossHead wrapper.
        # Let's use the wrapper's forward but we need to pass kwargs correctly.
        # Actually, let's just look at pretrain.py's train_batch to see how it calls it.
        # train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
        
        new_carry, loss, metrics, outputs, halted = model(
            carry=carry,
            batch=batch,
            return_keys=["logits", "preds"]
        )

    # Visualize
    preds = outputs["preds"].cpu().numpy()[0]
    targets = labels.cpu().numpy()[0]
    in_seq = inputs.cpu().numpy()[0]
    
    # The sequence is [Clues...Clues, Grid...Grid]
    # For 10x10, seq_len is 200. First 100 are clues, last 100 are grid.
    # Actually, let's check build_nonogram_dataset.py.
    # It concatenates clues and grid.
    
    clues_len = 100
    grid_len = 100
    
    input_clues = in_seq[:clues_len]
    target_grid = targets[clues_len:] # Targets are shifted? 
    # In build_nonogram_dataset:
    # inputs = np.concatenate([clues_flat, grid_flat])
    # labels = np.concatenate([clues_flat, grid_flat]) 
    # Wait, usually labels are shifted or masked.
    # Let's check build_nonogram_dataset.py again.
    # labels = np.full_like(inputs, -100)
    # labels[len(clues_flat):] = grid_flat
    # So labels has -100 for clues, and grid for the rest.
    
    print(f"Seq Len: {metadata.seq_len}")
    print(f"Inputs shape: {in_seq.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Preds shape: {preds.shape}")
    
    print(f"Preds stats - Min: {preds.min()}, Max: {preds.max()}, Mean: {preds.mean()}")
    print(f"Targets stats - Min: {targets.min()}, Max: {targets.max()}, Mean: {targets.mean()}")
    
    pred_grid = preds[clues_len:]
    real_target_grid = targets[clues_len:]

    print("\n--- Inference Result ---")
    
    print("\nTarget Grid:")
    print_grid(real_target_grid)
    
    print("\nPredicted Grid:")
    print_grid(pred_grid)
    
    # Calculate accuracy
    acc = np.mean(pred_grid == real_target_grid)
    print(f"\nGrid Accuracy: {acc * 100:.2f}%")

def print_grid(flat_grid, size=10):
    for i in range(size):
        row = flat_grid[i*size : (i+1)*size]
        print(" ".join(["■" if x == 1 else "□" for x in row]))

if __name__ == "__main__":
    main()

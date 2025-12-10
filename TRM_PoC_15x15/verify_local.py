import torch
import yaml
import os
import numpy as np
from omegaconf import OmegaConf
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from dataset.simple_nonogram_dataset import SimpleNonogramDataset

def verify_model():
    print("Searching for checkpoints...")
    checkpoint_dir = "checkpoints"
    
    # 1. Find Checkpoint
    found_checkpoints = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.startswith("step_"):
                found_checkpoints.append(os.path.join(root, file))
    
    if not found_checkpoints:
        print("No checkpoint found in 'checkpoints' directory.")
        return

    # Filter for 'versed-beluga' if possible, as that matches the user's download
    beluga_checkpoints = [cp for cp in found_checkpoints if "versed-beluga" in cp]
    if beluga_checkpoints:
        found_checkpoints = beluga_checkpoints

    # Sort by step number
    try:
        found_checkpoints.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    except:
        pass
        
    checkpoint_path = found_checkpoints[-1]
    checkpoint_dir_path = os.path.dirname(checkpoint_path)
    print(f"Found latest checkpoint: {checkpoint_path}")

    # 2. Load Config
    config_path = os.path.join(checkpoint_dir_path, "all_config.yaml")
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConf.create(config_dict)
    else:
        print(f"Config not found at {config_path}, falling back to local default config/nonogram_10x10_full.yaml")
        local_config_path = "config/nonogram_10x10_full.yaml"
        if os.path.exists(local_config_path):
             with open(local_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
             config = OmegaConf.create(config_dict)
        else:
            print("No config found. Exiting.")
            return

    # 3. Initialize Model
    arch_config = config.arch
    
    # Create config dict
    model_config_dict = dict(
        batch_size=1,
        seq_len=200,
        num_puzzle_identifiers=1,
        H_layers=1, # ignored
        pos_encodings=getattr(arch_config, 'pos_encodings', 'sinusoidal'),
        halt_max_steps=getattr(arch_config, 'halt_max_steps', 50),
        halt_exploration_prob=0.0,
        
        hidden_size=arch_config.hidden_size,
        num_heads=arch_config.num_heads,
        expansion=arch_config.expansion,
        puzzle_emb_ndim=arch_config.puzzle_emb_ndim,
        puzzle_emb_len=arch_config.puzzle_emb_len,
        H_cycles=arch_config.H_cycles,
        L_cycles=arch_config.L_cycles,
        L_layers=arch_config.L_layers,
        vocab_size=32,
        forward_dtype=arch_config.forward_dtype,
        loss_head=arch_config.loss.name if hasattr(arch_config.loss, 'name') else 'softmax_cross_entropy'
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(model_config_dict)
    
    # 4. Load State Dict
    print("Loading state dict...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Strip 'model.' prefix if present (from TrainState or DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 5. Load Data
    data_path = "dataset/nonogram_10x10_full"
    if not os.path.exists(data_path):
        # Fallback to poc if full not found (though user should have full)
        data_path = "dataset/nonogram_10x10_poc"
        
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    print(f"Loading test data from {data_path}...")
    dataset = SimpleNonogramDataset(data_path, split="test")
    
    # 6. Inference
    print("\nRunning Inference on 5 samples...")
    for i in range(5):
        sample = dataset[i]
        # Create batch dict
        batch = {
            "inputs": sample["inputs"].unsqueeze(0), # (1, SeqLen)
            "labels": sample["labels"].unsqueeze(0),
            "puzzle_identifiers": sample["puzzle_identifiers"].unsqueeze(0)
        }
        
        with torch.no_grad():
            # Initialize carry
            carry = model.initial_carry(batch)
            
            # Run inference loop
            while True:
                carry, outputs = model(carry=carry, batch=batch)
                if carry.halted.all():
                    break
            
            logits = outputs["logits"]
            
        preds = torch.argmax(logits, dim=-1)
        
        # Extract grid (last 100 tokens)
        grid_pred = preds[0, 100:].numpy()
        grid_true = batch["labels"][0, 100:].numpy()
        
        correct = (grid_pred == grid_true).sum()
        total = 100
        print(f"Sample {i}: Accuracy {correct}/{total}")
        if correct < total:
            print("  Pred:", grid_pred)
            print("  True:", grid_true)
        else:
            print("  Perfect Match!")

if __name__ == "__main__":
    verify_model()

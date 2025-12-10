import torch
import time
import numpy as np
import os
import yaml
from omegaconf import OmegaConf
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from dataset.simple_nonogram_dataset import SimpleNonogramDataset

def load_model_and_data():
    # 1. Find Checkpoint
    checkpoint_dir = "checkpoints"
    found_checkpoints = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.startswith("step_"):
                found_checkpoints.append(os.path.join(root, file))
    
    # Filter for 'versed-beluga'
    beluga_checkpoints = [cp for cp in found_checkpoints if "versed-beluga" in cp]
    if beluga_checkpoints:
        found_checkpoints = beluga_checkpoints
    
    found_checkpoints.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    checkpoint_path = found_checkpoints[-1]
    checkpoint_dir_path = os.path.dirname(checkpoint_path)
    
    # 2. Load Config
    config_path = os.path.join(checkpoint_dir_path, "all_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConf.create(config_dict)
    else:
        # Fallback
        with open("config/nonogram_10x10_full.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConf.create(config_dict)

    # 3. Initialize Model
    arch_config = config.arch
    model_config_dict = dict(
        batch_size=1,
        seq_len=200,
        num_puzzle_identifiers=1,
        H_layers=1, 
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
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 5. Load Data
    data_path = "dataset/nonogram_10x10_full"
    if not os.path.exists(data_path):
        data_path = "dataset/nonogram_10x10_poc"
    dataset = SimpleNonogramDataset(data_path, split="test")
    
    return model, dataset

def measure_speed():
    model, dataset = load_model_and_data()
    
    num_samples = 100
    print(f"Measuring inference speed on {num_samples} samples (CPU)...")
    
    latencies = []
    
    # Warmup
    print("Warming up...")
    for i in range(10):
        sample = dataset[i]
        batch = {
            "inputs": sample["inputs"].unsqueeze(0),
            "labels": sample["labels"].unsqueeze(0),
            "puzzle_identifiers": sample["puzzle_identifiers"].unsqueeze(0)
        }
        with torch.no_grad():
            carry = model.initial_carry(batch)
            while True:
                carry, outputs = model(carry=carry, batch=batch)
                if carry.halted.all():
                    break
    
    print("Running measurement...")
    for i in range(num_samples):
        sample = dataset[i]
        batch = {
            "inputs": sample["inputs"].unsqueeze(0),
            "labels": sample["labels"].unsqueeze(0),
            "puzzle_identifiers": sample["puzzle_identifiers"].unsqueeze(0)
        }
        
        start_time = time.perf_counter()
        with torch.no_grad():
            carry = model.initial_carry(batch)
            while True:
                carry, outputs = model(carry=carry, batch=batch)
                if carry.halted.all():
                    break
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000) # Convert to ms
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{num_samples}")

    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    print("\n--- Inference Speed Results (CPU) ---")
    print(f"Mean Time:   {mean_latency:.2f} ms")
    print(f"Median Time: {median_latency:.2f} ms")
    print(f"Std Dev:     {std_latency:.2f} ms")
    print(f"Min Time:    {min_latency:.2f} ms")
    print(f"Max Time:    {max_latency:.2f} ms")

if __name__ == "__main__":
    measure_speed()

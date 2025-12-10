import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from omegaconf import OmegaConf
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from dataset.simple_nonogram_dataset import SimpleNonogramDataset
import tqdm

def get_model_config(checkpoint_dir_path):
    config_path = os.path.join(checkpoint_dir_path, "all_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConf.create(config_dict)
    else:
        with open("config/nonogram_10x10_full.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConf.create(config_dict)
    return config

def init_model(config):
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
    return TinyRecursiveReasoningModel_ACTV1(model_config_dict)

def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def plot_curve():
    # 1. Find Checkpoints
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
    
    # 2. Load Data
    data_path = "dataset/nonogram_10x10_full"
    if not os.path.exists(data_path):
        data_path = "dataset/nonogram_10x10_poc"
    dataset = SimpleNonogramDataset(data_path, split="test")
    
    # Use a subset for speed
    num_eval_samples = 100
    indices = range(num_eval_samples)
    
    steps = []
    accuracies = []
    
    checkpoint_dir_path = os.path.dirname(found_checkpoints[0])
    config = get_model_config(checkpoint_dir_path)
    model = init_model(config)
    
    print(f"Evaluating {len(found_checkpoints)} checkpoints on {num_eval_samples} samples...")
    
    for cp_path in found_checkpoints:
        step_num = int(os.path.basename(cp_path).split("_")[1])
        model = load_checkpoint(model, cp_path)
        
        correct_count = 0
        for i in indices:
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
                logits = outputs["logits"]
            
            preds = torch.argmax(logits, dim=-1)
            grid_pred = preds[0, 100:].numpy()
            grid_true = batch["labels"][0, 100:].numpy()
            
            if (grid_pred == grid_true).all():
                correct_count += 1
        
        acc = correct_count / num_eval_samples
        steps.append(step_num)
        accuracies.append(acc)
        print(f"Step {step_num}: Accuracy {acc:.2f}")

    # 3. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, accuracies, marker='o', linestyle='-', color='#e74c3c', linewidth=2, markersize=6)
    plt.title("Test Exact Match Accuracy vs Training Steps", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Exact Match Accuracy", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1.05)
    
    # Add labels
    for i, txt in enumerate(accuracies):
        plt.annotate(f"{txt:.2f}", (steps[i], accuracies[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    filename = "training_curve.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved curve to {filename}")

if __name__ == "__main__":
    plot_curve()

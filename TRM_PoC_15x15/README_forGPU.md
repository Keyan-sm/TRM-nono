# TRM Training Instructions

## Prerequisites
- Python 3.9+
- CUDA-capable GPU (NVIDIA) or Mac with MPS (Apple Silicon)
- 16GB+ RAM

## Setup
1. Unzip this folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training
To run the full training on the 10x10 dataset:

```bash
python3 pretrain.py --config-name nonogram_10x10_full
```

## Cloud GPU Setup (e.g., Lambda, Vast.ai, RunPod)
1. **Rent an Instance:** Look for NVIDIA A100, A6000, or RTX 4090. Ubuntu 20.04/22.04 is standard.
2. **Transfer Files:**
   - Use `scp` or drag-and-drop if the provider has a web terminal.
   - Example: `scp TRM_PoC_10x10_Friend_Package.zip root@<ip_address>:/root/`
3. **Install System Dependencies:**
   ```bash
   apt-get update && apt-get install -y unzip python3-pip
   ```
4. **Unzip and Install:**
   ```bash
   unzip TRM_PoC_10x10_Friend_Package.zip
   cd TRM_PoC_10x10
   pip install -r requirements.txt
   ```
5. **Run Training:**
   ```bash
   python3 pretrain.py --config-name nonogram_10x10_full
   ```
   *Tip: Use `tmux` or `nohup` so training doesn't stop if you disconnect.*

## Configuration
The configuration is located in `config/nonogram_10x10_full.yaml`.
- `global_batch_size`: Adjust this if you run out of memory (e.g., lower to 64 or 32).
- `forward_dtype`: `float16` is set for speed. Change to `float32` if you encounter stability issues (NaN loss).

## Monitoring
The script will print metrics to the console. You can also enable `wandb` in `pretrain.py` if you have an account.

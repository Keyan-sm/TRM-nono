# Vast.ai Training Guide

Since you are running this yourself, here is the exact workflow. You do **not** need to give me your login. You will control the remote machine from your terminal.

## 1. Setup Vast.ai
1.  **Create Account:** Go to [vast.ai](https://vast.ai/) and sign up.
2.  **Add Credit:** Add ~$5-10. This will last for many hours of training.
3.  **Rent Instance:**
    *   Go to "Client" -> "Create".
    *   **Filters:**
        *   GPU: RTX 3090 or RTX 4090 (Best value).
        *   Disk Space: 32GB+ (to be safe).
        *   Image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` (or similar default PyTorch image).
    *   Click **Rent** on a machine with a good reliability score (>95%).

## 1.5. Setup SSH Key (First Time Only)
If you see an error about "SSH keys not set up":
1.  **On your Mac Terminal**, run: `ssh-keygen -t rsa` (Press Enter for all prompts).
2.  Copy the key: `cat ~/.ssh/id_rsa.pub | pbcopy`
3.  **On Vast.ai**: Go to **Account** (left sidebar) -> **SSH Keys**.
4.  Paste the key and click **Add**.
5.  **Important:** You must **Destroy** your current instance and **Rent a new one** for the key to work.

## 2. Connect & Transfer Files
Once the instance is "Running", click the **Connect** button to see the SSH commands.

### 1. Upload the Project
Open a terminal on your **local machine** (where the zip file is) and run:
```bash
# Upload the 15x15 package
scp -P 31871 TRM_PoC_15x15/TRM_PoC_15x15_Package.zip root@59.26.209.89:/root/
```

### 2. Connect to the Instance
```bash
ssh -p 31871 root@59.26.209.89
```

### 3. Setup and Train (On the Remote Machine)
Once logged in:
```bash
# Unzip
unzip -o TRM_PoC_15x15_Package.zip
cd TRM_PoC_15x15

# Generate Dataset (Crucial Step!)
bash setup_dataset.sh

# Start Training (Background)
# This uses nohup so it keeps running if you disconnect
nohup python3 pretrain.py --config-name nonogram_15x15 > training.log 2>&1 &

# Check if it's running
tail -f training.log
```
    *   Check progress: `tail -f training.log`
    *   Check GPU usage: `nvidia-smi`

## 4. Download Results
When finished (check `training.log`):
1.  **Exit SSH:** Type `exit`.
2.  **Download Checkpoints:**
    ```bash
    scp -P <PORT> -r root@<IP>:/root/TRM_PoC_10x10/checkpoints .
    ```
3.  **Stop Instance:** Go to Vast.ai and **Stop/Destroy** the instance so you stop paying!

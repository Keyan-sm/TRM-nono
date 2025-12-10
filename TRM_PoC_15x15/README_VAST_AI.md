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

### Step A: Transfer the Code
Run this command from **your local terminal** (where the zip file is):

```bash
# Replace <PORT> and <IP> with the values from Vast.ai (e.g., -P 12345 root@1.2.3.4)
scp -P <PORT> TRM_PoC_10x10_Friend_Package.zip root@<IP>:/root/
```

### Step B: Connect to the Instance
Run the SSH command provided by Vast.ai:

```bash
ssh -p <PORT> root@<IP>
```

## 3. Run Training (On the Remote Machine)
Once you are logged in via SSH:

1.  **Unzip and Install:**
    ```bash
    apt-get update && apt-get install -y unzip
    unzip TRM_PoC_10x10_Friend_Package.zip
    cd TRM_PoC_10x10
    pip install -r requirements.txt
    ```

2.  **Start Training:**
    ```bash
    # Use nohup so it keeps running if you disconnect
    nohup python3 pretrain.py --config-name nonogram_10x10_full > training.log 2>&1 &
    ```

3.  **Monitor:**
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

# TRM 15x15 Nonogram Training

This package contains the code and dataset for training TRM on 15x15 Nonograms.

## Setup on Vast.ai

1.  Upload the package:
    ```bash
    scp -P <PORT> TRM_PoC_15x15_Package.zip root@<IP>:/root/
    ```

2.  SSH into the instance:
    ```bash
    ssh -p <PORT> root@<IP>
    ```

3.  Unzip:
    ```bash
    unzip TRM_PoC_15x15_Package.zip
    cd TRM_PoC_15x15
    ```

4.  Install dependencies (if not already installed):
    ```bash
    pip install -r requirements.txt
    ```

5.  Run training:
    ```bash
    python3 pretrain.py --config-name nonogram_15x15
    ```

## Configuration

The configuration is in `config/nonogram_15x15.yaml`.
-   Batch size: 32 (adjusted for 15x15)
-   Epochs: 50
-   Model: TRM (Tiny Recursive Model)

## Dataset

The dataset is already processed and located in `dataset/nonogram_15x15`.
It contains 661k training samples and 15k test samples.

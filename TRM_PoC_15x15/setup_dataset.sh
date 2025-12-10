#!/bin/bash
echo "Building 15x15 Dataset..."
python3 dataset/build_dataset_15x15.py --input_dir dataset/raw_15x15 --output_dir dataset

# Move to correct folder structure expected by config
mkdir -p dataset/nonogram_15x15
mv dataset/train dataset/nonogram_15x15/
mv dataset/test dataset/nonogram_15x15/
echo "Dataset built successfully!"

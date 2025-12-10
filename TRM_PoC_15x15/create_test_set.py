import os
import shutil
import json

base_dir = "dataset/nonogram_10x10_poc"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Copy directory
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
shutil.copytree(train_dir, test_dir)

# Rename files
for filename in os.listdir(test_dir):
    if filename.startswith("train__"):
        new_filename = filename.replace("train__", "test__")
        os.rename(os.path.join(test_dir, filename), os.path.join(test_dir, new_filename))

# Update dataset.json
json_path = os.path.join(test_dir, "dataset.json")
with open(json_path, "r") as f:
    data = json.load(f)

data["sets"] = ["test"]

with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print("Created test dataset from train dataset.")

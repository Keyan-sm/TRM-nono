import os
import shutil
import zipfile

source_dir = "."
output_filename = "TRM_PoC_15x15_Package.zip"

def zip_dir(path, ziph):
    for root, dirs, files in os.walk(path):
        # Exclude hidden directories and specific folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['wandb', 'checkpoints', '__pycache__', 'outputs', 'raw_15x15', 'nonogram_10x10']]
        
        for file in files:
            if file.startswith('.') or file.endswith('.zip') or file.endswith('.DS_Store'):
                continue
            
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, os.path.join(path, '..'))
            ziph.write(file_path, arcname)

print(f"Creating {output_filename}...")
with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zip_dir(source_dir, zipf)

print(f"Done. Package created at {os.path.abspath(output_filename)}")

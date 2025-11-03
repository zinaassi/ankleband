import pandas as pd
import os

# Check if files exist
dataset_path = "data/dataset"
print(f"Checking dataset folder: {dataset_path}")
print(f"Folder exists: {os.path.exists(dataset_path)}")

files = os.listdir(dataset_path)
print(f"\nFound {len(files)} files:")
for f in files:
    print(f"  - {f}")

# Try to load one file
print(f"\n\nTrying to load first file...")
first_file = files[0]
df = pd.read_hdf(os.path.join(dataset_path, first_file))
print(f"✅ Successfully loaded {first_file}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
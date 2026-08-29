from pathlib import Path
import pandas as pd

# -------------------------------------------------------
# Locate project root
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

df = pd.read_csv(dataset_path)

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)

print(f"\nDataset Path : {dataset_path}")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Samples:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())
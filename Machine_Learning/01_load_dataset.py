"""
====================================================================
TinyML Workshop
Part V : Classical Machine Learning for TinyML Deployment

Slide 4 : Loading the Processed Dataset
====================================================================

Objective:
    Load the processed dataset generated in Part IV and verify that it
    is ready for Classical Machine Learning.

Author : Dr. C. Sivananda Reddy
====================================================================
"""

from pathlib import Path
import pandas as pd


# ==============================================================
# Locate the dataset
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"


# ==============================================================
# Load Dataset
# ==============================================================

print("=" * 60)
print("Loading Processed Dataset...")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
    print("\nDataset loaded successfully.\n")

except FileNotFoundError:
    print(f"\nERROR: Dataset not found at:\n{DATASET_PATH}")
    exit()


# ==============================================================
# Display Dataset Information
# ==============================================================

print("=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("Column Names")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("First Five Records")
print("=" * 60)
print(df.head())


# ==============================================================
# Check Missing Values
# ==============================================================

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)
print(df.isnull().sum())


# ==============================================================
# Separate Features and Target
# ==============================================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\n" + "=" * 60)
print("Feature Matrix Shape")
print("=" * 60)
print(X.shape)

print("\n" + "=" * 60)
print("Target Vector Shape")
print("=" * 60)
print(y.shape)

print("\n" + "=" * 60)
print("Target Classes")
print("=" * 60)
print(y.unique())


# ==============================================================
# Summary
# ==============================================================

print("\n" + "=" * 60)
print("Dataset Verification Completed Successfully")
print("=" * 60)

print(f"Number of Samples  : {len(df)}")
print(f"Number of Features : {X.shape[1]}")
print(f"Number of Classes  : {len(y.unique())}")

print("\nThe processed dataset is ready for Classical Machine Learning.")
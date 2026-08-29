"""
====================================================================
TinyML Workshop
Part V : Classical Machine Learning for TinyML Deployment

Slide 5 : Splitting the Dataset into Training and Testing Sets
====================================================================
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==============================================================
# Locate Dataset
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
    print(f"\nERROR : Dataset not found.\n{DATASET_PATH}")
    exit()


# ==============================================================
# Separate Features and Labels
# ==============================================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("=" * 60)
print("Feature Matrix and Target Vector")
print("=" * 60)

print(f"Feature Matrix Shape : {X.shape}")
print(f"Target Vector Shape  : {y.shape}")


# ==============================================================
# Train-Test Split
# ==============================================================

print("\n" + "=" * 60)
print("Splitting Dataset")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
    shuffle=True
)


# ==============================================================
# Display Shapes
# ==============================================================

print("\nTraining Dataset")
print("-" * 60)
print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")

print("\nTesting Dataset")
print("-" * 60)
print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")


# ==============================================================
# Class Distribution
# ==============================================================

print("\n" + "=" * 60)
print("Training Class Distribution")
print("=" * 60)
print(y_train.value_counts())

print("\n" + "=" * 60)
print("Testing Class Distribution")
print("=" * 60)
print(y_test.value_counts())


# ==============================================================
# Summary
# ==============================================================

print("\n" + "=" * 60)
print("Dataset Successfully Prepared for Model Training")
print("=" * 60)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

print("\nDataset is ready for machine learning model training.")
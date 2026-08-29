from pathlib import Path
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# Fix Random Seeds
# -------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------
# Locate Dataset
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv(dataset_path)

# -------------------------------------------------------
# Separate Features and Labels
# -------------------------------------------------------

X = df.drop(columns=["Label"])

y = df["Label"]

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# -------------------------------------------------------
# Display Results
# -------------------------------------------------------

print("=" * 60)
print("Train-Test Split Completed")
print("=" * 60)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("\nTraining Shape")
print(X_train.shape)

print("\nTesting Shape")
print(X_test.shape)

print("\nTraining Label Distribution")
print(y_train.value_counts().sort_index().to_string())

print("\nTesting Label Distribution")
print(y_test.value_counts().sort_index().to_string())

# -------------------------------------------------------
# Save Split Datasets
# -------------------------------------------------------

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nTraining and Testing datasets saved successfully.")
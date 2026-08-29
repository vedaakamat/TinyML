from pathlib import Path
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from build_mlp import MLP


# -------------------------------------------------------
# Fix Random Seeds
# -------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------------
# Locate Project and Train-Test CSV Files
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(__file__).resolve().parent

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH  = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH  = DATA_DIR / "y_test.csv"


# -------------------------------------------------------
# Check Required Files
# -------------------------------------------------------

required_files = [
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# -------------------------------------------------------
# Load Previously Created Train-Test Data
# -------------------------------------------------------

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH)
y_test = pd.read_csv(Y_TEST_PATH)


# -------------------------------------------------------
# Convert DataFrames to NumPy Arrays
# -------------------------------------------------------

X_train = X_train.values
X_test = X_test.values

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


# -------------------------------------------------------
# Display Loaded Dataset Information
# -------------------------------------------------------

print("=" * 60)
print("Previously Created Train-Test Data Loaded")
print("=" * 60)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("\nTraining Shape")
print(X_train.shape)

print("\nTesting Shape")
print(X_test.shape)

print("\nTraining Label Distribution")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTesting Label Distribution")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
    .to_string()
)


# -------------------------------------------------------
# Convert Training Data to PyTorch Tensors
# -------------------------------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)


# -------------------------------------------------------
# Convert Testing Data to PyTorch Tensors
# -------------------------------------------------------

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# -------------------------------------------------------
# Dataset and DataLoader
# -------------------------------------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# -------------------------------------------------------
# Model
# -------------------------------------------------------

model = MLP()


# -------------------------------------------------------
# Loss Function
# -------------------------------------------------------

criterion = nn.CrossEntropyLoss()


# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------------------------------------
# Training Configuration
# -------------------------------------------------------

NUM_EPOCHS = 20


print("\n" + "=" * 60)
print("Training Pipeline Configured Successfully")
print("=" * 60)

print(f"\nTraining Samples : {len(train_dataset)}")
print(f"Testing Samples  : {len(test_dataset)}")
print(f"Mini-batch Size  : 32")
print(f"Total Batches    : {len(train_loader)}")

print("\nLoss Function")
print(criterion)

print("\nOptimizer Configuration")
print(f"Optimizer      : Adam")
print(f"Learning Rate  : {optimizer.param_groups[0]['lr']}")
print(f"Batch Size     : {train_loader.batch_size}")
print(f"Epochs         : {NUM_EPOCHS}")

print("\nTraining Configuration")
print("-" * 40)
print(f"Input Features     : 6")
print(f"Output Classes     : 5")
print(f"Hidden Layers      : 64 -> 32")
print(f"Training Samples   : {len(train_dataset)}")
print(f"Testing Samples    : {len(test_dataset)}")
print(f"Batch Size         : 32")
print(f"Epochs             : {NUM_EPOCHS}")
print(f"Learning Rate      : 0.001")
print(f"Loss Function      : CrossEntropyLoss")
print(f"Optimizer          : Adam")


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------

print("\nStarting Training...\n")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for inputs, labels in train_loader:

        # Forward Pass
        outputs = model(inputs)

        # Compute Loss
        loss = criterion(outputs, labels)

        # Clear Previous Gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
        f"Loss = {epoch_loss:.6f}"
    )


print("\nTraining Completed Successfully.")


# -------------------------------------------------------
# Save Trained Model
# -------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "Models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "mlp_model.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n" + "=" * 60)
print("Model Saved Successfully")
print("=" * 60)

print(f"\nModel Path : {MODEL_PATH}")
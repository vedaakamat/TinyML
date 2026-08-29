from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from build_mlp import MLP


# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Previously created test data
X_TEST_PATH = Path(__file__).resolve().parent / "X_test.csv"
Y_TEST_PATH = Path(__file__).resolve().parent / "y_test.csv"

# Trained model
MODEL_PATH = PROJECT_ROOT / "Models" / "mlp_model.pth"


# -------------------------------------------------------
# Check Required Files
# -------------------------------------------------------

if not X_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test feature file not found: {X_TEST_PATH}"
    )

if not Y_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test label file not found: {Y_TEST_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


# -------------------------------------------------------
# Load Previously Created Test Dataset
# -------------------------------------------------------

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)


# -------------------------------------------------------
# Convert to NumPy Arrays
# -------------------------------------------------------

X_test = X_test.values

y_test = y_test.values.ravel()


# -------------------------------------------------------
# Display Test Dataset Information
# -------------------------------------------------------

print("=" * 60)
print("Previously Created Test Dataset Loaded")
print("=" * 60)

print(f"\nTesting Samples : {len(X_test)}")
print(f"Testing Shape   : {X_test.shape}")

print("\nTesting Label Distribution")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
    .to_string()
)


# -------------------------------------------------------
# Convert to Tensor
# -------------------------------------------------------

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
)


# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model = MLP()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
    )
)

model.eval()


# -------------------------------------------------------
# Inference
# -------------------------------------------------------

with torch.no_grad():

    outputs = model(X_test_tensor)

    predictions = torch.argmax(
        outputs,
        dim=1,
    ).numpy()


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
)

cm = confusion_matrix(
    y_test,
    predictions,
)


# -------------------------------------------------------
# Display Results
# -------------------------------------------------------

print("\n" + "=" * 60)
print("PyTorch FP32 Validation")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions,
    )
)
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEEP_LEARNING_DIR = PROJECT_ROOT / "Deep_Learning"
CALIBRATION_DIR = PROJECT_ROOT / "Calibration"

CALIBRATION_DIR.mkdir(exist_ok=True)

X_TRAIN_FILE = DEEP_LEARNING_DIR / "X_train.csv"
CALIBRATION_FILE = CALIBRATION_DIR / "calibration_data.npy"

# --------------------------------------------------
# Load Training Data
# --------------------------------------------------

print("=" * 60)
print("Preparing Calibration Dataset")
print("=" * 60)

X_train = pd.read_csv(X_TRAIN_FILE)

print(f"\nTraining Samples  : {len(X_train)}")
print(f"Feature Dimension : {X_train.shape[1]}")

# --------------------------------------------------
# Select Representative Calibration Samples
# --------------------------------------------------

NUM_CALIBRATION_SAMPLES = 500

if len(X_train) < NUM_CALIBRATION_SAMPLES:
    raise ValueError(
        f"Training set contains only {len(X_train)} samples."
    )

calibration_data = (
    X_train
    .sample(n=NUM_CALIBRATION_SAMPLES, random_state=42)
    .to_numpy(dtype=np.float32)
)

# --------------------------------------------------
# Save Calibration Dataset
# --------------------------------------------------

np.save(CALIBRATION_FILE, calibration_data)

# --------------------------------------------------
# Display Summary
# --------------------------------------------------

print("\nCalibration Dataset Created Successfully")
print(f"Calibration Samples : {calibration_data.shape[0]}")
print(f"Feature Dimension   : {calibration_data.shape[1]}")

print(f"\nSaved to:\n{CALIBRATION_FILE}")

print("\nDone.")
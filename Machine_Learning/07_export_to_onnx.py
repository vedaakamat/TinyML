"""
====================================================================
TinyML Workshop
Part V : Classical Machine Learning for TinyML Deployment

Slide 10 : Exporting the Selected Model to ONNX
====================================================================
"""

from pathlib import Path

import joblib
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ==============================================================
# Project Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

MODELS_DIR = PROJECT_ROOT / "Models"
ONNX_DIR = PROJECT_ROOT / "ONNX"

ONNX_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
ONNX_MODEL_PATH = ONNX_DIR / "random_forest_model.onnx"

# ==============================================================
# Load Dataset
# ==============================================================

print("=" * 60)
print("Loading Processed Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.\n")

X = df.iloc[:, :-1]

# ==============================================================
# Load Selected Model
# ==============================================================

print("=" * 60)
print("Loading Selected Model...")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.\n")

# ==============================================================
# Convert to ONNX
# ==============================================================

print("=" * 60)
print("Converting Model to ONNX...")
print("=" * 60)

initial_type = [
    (
        "float_input",
        FloatTensorType([None, X.shape[1]])
    )
]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type
)

# ==============================================================
# Save ONNX Model
# ==============================================================

with open(ONNX_MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model exported successfully.\n")

print("=" * 60)
print("Generated Files")
print("=" * 60)

print(f"Model      : {MODEL_PATH}")
print(f"ONNX Model : {ONNX_MODEL_PATH}")

print("\nModel is ready for ONNX Runtime validation.")
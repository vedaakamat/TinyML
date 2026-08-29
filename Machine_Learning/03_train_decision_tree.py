"""
====================================================================
TinyML Workshop
Part V : Classical Machine Learning for TinyML Deployment

Slide 6 : Training a Decision Tree Classifier
====================================================================
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ==============================================================
# Project Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

MODELS_DIR = PROJECT_ROOT / "Models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "decision_tree_model.pkl"

# ==============================================================
# Load Dataset
# ==============================================================

print("=" * 60)
print("Loading Processed Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.\n")

# ==============================================================
# Separate Features and Labels
# ==============================================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ==============================================================
# Train-Test Split
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
    shuffle=True
)

# ==============================================================
# Create Decision Tree Model
# ==============================================================

print("=" * 60)
print("Creating Decision Tree Classifier...")
print("=" * 60)

model = DecisionTreeClassifier(
    random_state=42
)

# ==============================================================
# Train Model
# ==============================================================

print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

print("Model training completed successfully.\n")

# ==============================================================
# Save Model
# ==============================================================

joblib.dump(model, MODEL_PATH)

print("=" * 60)
print("Model Saved Successfully")
print("=" * 60)

print(f"Model File : {MODEL_PATH}")

print("\nDecision Tree model is ready for evaluation.")
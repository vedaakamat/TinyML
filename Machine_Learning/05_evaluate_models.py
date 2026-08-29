"""
====================================================================
TinyML Workshop
Part V : Classical Machine Learning for TinyML Deployment

Slide 8 : Evaluating Machine Learning Models
====================================================================
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

# ==============================================================
# Project Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

MODELS_DIR = PROJECT_ROOT / "Models"

DT_MODEL = MODELS_DIR / "decision_tree_model.pkl"
RF_MODEL = MODELS_DIR / "random_forest_model.pkl"

# ==============================================================
# Load Dataset
# ==============================================================

print("=" * 60)
print("Loading Processed Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.\n")

# ==============================================================
# Features and Labels
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
# Load Models
# ==============================================================

print("=" * 60)
print("Loading Trained Models...")
print("=" * 60)

decision_tree = joblib.load(DT_MODEL)
random_forest = joblib.load(RF_MODEL)

print("Models loaded successfully.\n")

# ==============================================================
# Evaluate Function
# ==============================================================

def evaluate_model(model, model_name):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")

    print(cm)

# ==============================================================
# Evaluate Both Models
# ==============================================================

evaluate_model(
    decision_tree,
    "Decision Tree Results"
)

print()

evaluate_model(
    random_forest,
    "Random Forest Results"
)

print("\nModel evaluation completed successfully.")
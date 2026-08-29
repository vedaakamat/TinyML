"""
====================================================================
TinyML Workshop
Part V : Classical Machine Learning for TinyML Deployment

Slide 9 : Comparing Models and Selecting the Best Model
====================================================================
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

df = pd.read_csv(DATASET_PATH)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

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

decision_tree = joblib.load(DT_MODEL)
random_forest = joblib.load(RF_MODEL)


# ==============================================================
# Evaluation Function
# ==============================================================

def evaluate(model):

    prediction = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(
            y_test,
            prediction,
            average="weighted"
        ),
        "Recall": recall_score(
            y_test,
            prediction,
            average="weighted"
        ),
        "F1 Score": f1_score(
            y_test,
            prediction,
            average="weighted"
        ),
    }


dt = evaluate(decision_tree)
rf = evaluate(random_forest)

comparison = pd.DataFrame(
    {
        "Decision Tree": dt,
        "Random Forest": rf
    }
)

comparison = comparison.round(4)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison)

print("\n")

if rf["Accuracy"] > dt["Accuracy"]:

    best_model = "Random Forest"

else:

    best_model = "Decision Tree"

print("=" * 70)
print(f"Selected Model : {best_model}")
print("=" * 70)

print("\nSelected model is ready for ONNX export.")
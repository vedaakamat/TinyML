import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Prepare X and y
X = df[["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]]
y = df["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display shapes
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
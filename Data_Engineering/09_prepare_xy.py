import pandas as pd

# Load dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Separate features and labels
X = df[["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]]
y = df["Label"]

# Display shapes
print("Feature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)

# Display first five rows
print("\nFeature Matrix:")
print(X.head())

print("\nTarget Labels:")
print(y.head())
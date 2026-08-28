import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Count samples in each class
class_counts = df["Label"].value_counts()

print(class_counts)

# Plot class distribution
plt.figure(figsize=(8,5))
class_counts.plot(kind="bar")

plt.title("Class Distribution")
plt.xlabel("Motion Class")
plt.ylabel("Number of Samples")

plt.grid(axis="y")
plt.show()
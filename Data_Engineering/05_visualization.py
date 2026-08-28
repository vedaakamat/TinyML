import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Plot accelerometer signals
plt.figure(figsize=(12,5))
plt.plot(df["Ax"], label="ax")
plt.plot(df["Ay"], label="ay")
plt.plot(df["Az"], label="az")

plt.title("Accelerometer Signals")
plt.xlabel("Sample Number")
plt.ylabel("Acceleration")
plt.legend()
plt.grid(True)

plt.show()
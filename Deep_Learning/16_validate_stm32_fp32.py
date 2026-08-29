"""
===========================================================
16_validate_stm32_fp32.py

Validate the deployed FP32 MLP model running on STM32 by
sending every sample from X_test.csv over UART and
recording the predicted class.

Author : Dr. C Sivananda Reddy
===========================================================
"""

from pathlib import Path
import time

import pandas as pd
import serial


# ==========================================================
# Serial Port Configuration
# ==========================================================

COM_PORT = "COM5"          # Change if required
BAUD_RATE = 115200
TIMEOUT = 2


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

X_TEST_PATH = PROJECT_ROOT / "Deep_Learning" / "X_test.csv"

OUTPUT_PATH = PROJECT_ROOT / "Deep_Learning" / "stm32_predictions.csv"


# ==========================================================
# Load Test Dataset
# ==========================================================

print("=" * 60)
print("STM32 FP32 Model Validation")
print("=" * 60)

X_test = pd.read_csv(X_TEST_PATH)

print(f"Test Samples : {len(X_test)}")


# ==========================================================
# Open Serial Port
# ==========================================================

print("\nOpening Serial Port...")

ser = serial.Serial(
    port=COM_PORT,
    baudrate=BAUD_RATE,
    timeout=TIMEOUT
)

# Give STM32 time to reset
time.sleep(2)

# Remove startup messages from UART buffer
ser.reset_input_buffer()

print("Connected to STM32.\n")


# ==========================================================
# Validate All Samples
# ==========================================================

predictions = []

print("Running Validation...\n")

for sample_index, row in X_test.iterrows():

    # ----------------------------------------------
    # Convert one sample into CSV format
    # ----------------------------------------------

    sample = ",".join(f"{value:.6f}" for value in row.values)

    # ----------------------------------------------
    # Send sample
    # ----------------------------------------------

    ser.write((sample + "\n").encode())

    # ----------------------------------------------
    # Receive prediction
    # ----------------------------------------------

    response = ser.readline().decode().strip()

    if response == "":

        print(f"Sample {sample_index + 1:4d} : No Response")

        predictions.append(-1)

        continue

    try:

        prediction = int(response)

        predictions.append(prediction)

        print(
            f"Sample {sample_index + 1:4d}/{len(X_test)}"
            f"   Prediction : {prediction}"
        )

    except ValueError:

        print(
            f"Sample {sample_index + 1:4d}"
            f"   Invalid Response : {response}"
        )

        predictions.append(-1)


# ==========================================================
# Close Serial Port
# ==========================================================

ser.close()


# ==========================================================
# Save Predictions
# ==========================================================

predictions_df = pd.DataFrame({
    "Prediction": predictions
})

predictions_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================================
# Summary
# ==========================================================

valid_predictions = sum(p >= 0 for p in predictions)

print("\n" + "=" * 60)
print("Validation Completed")
print("=" * 60)

print(f"Total Samples      : {len(predictions)}")
print(f"Valid Predictions  : {valid_predictions}")
print(f"Invalid Predictions: {len(predictions) - valid_predictions}")

print("\nPredictions saved to:")

print(OUTPUT_PATH)

print("=" * 60)
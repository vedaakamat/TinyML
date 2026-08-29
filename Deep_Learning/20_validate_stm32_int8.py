"""
==========================================================
File        : 20_validate_stm32_int8.py

Purpose     : Validate the deployed INT8 MLP model
              running on STM32.

Workflow
--------
1. Load X_test.csv (already normalized)
2. Convert to UINT8 (0-255)
3. Send each sample to STM32
4. Receive predicted class
5. Save predictions
==========================================================
"""

import time
import serial
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

SERIAL_PORT = "COM4"      # Change to your COM port
BAUD_RATE = 115200
TIMEOUT = 2

# ==========================================================
# Load Test Dataset
# ==========================================================

print("=" * 60)
print("STM32 INT8 Model Validation")
print("=" * 60)

print("\nLoading test dataset...")

X_test = pd.read_csv("X_test.csv")

print(f"Number of Test Samples : {len(X_test)}")

# ==========================================================
# Convert Normalized Features to UINT8
# ==========================================================

print("\nConverting normalized features to UINT8...")

X_test_quantized = (X_test.values * 255.0)

X_test_quantized = X_test_quantized.round()

X_test_quantized = X_test_quantized.clip(0, 255)

X_test_quantized = X_test_quantized.astype("uint8")

print("Conversion completed.")

# ==========================================================
# Open Serial Port
# ==========================================================

print("\nOpening serial port...")

ser = serial.Serial(
    port=SERIAL_PORT,
    baudrate=BAUD_RATE,
    timeout=TIMEOUT
)

# Wait for STM32 reset
time.sleep(2)

# Clear startup messages
ser.reset_input_buffer()

print("STM32 Connected.")

# ==========================================================
# Run Validation
# ==========================================================

predictions = []

total_samples = len(X_test_quantized)

print("\nRunning inference on STM32...\n")

for index, sample in enumerate(X_test_quantized):

    # ---------------------------------------------
    # Convert one feature vector into CSV string
    # ---------------------------------------------

    tx = ",".join(str(int(value)) for value in sample)

    tx += "\n"

    # ---------------------------------------------
    # Send sample
    # ---------------------------------------------

    ser.write(tx.encode())

    # ---------------------------------------------
    # Receive prediction
    # ---------------------------------------------

    while True:

        line = ser.readline().decode(
            errors="ignore"
        ).strip()

        if line == "":
            continue

        try:

            prediction = int(line)

            predictions.append(prediction)

            break

        except ValueError:

            # Ignore any banner/status messages
            continue

    # ---------------------------------------------
    # Progress
    # ---------------------------------------------

    if (index + 1) % 100 == 0 or (index + 1) == total_samples:

        print(f"Processed {index + 1}/{total_samples}")

# ==========================================================
# Close Serial Port
# ==========================================================

ser.close()

print("\nInference completed.")

# ==========================================================
# Save Predictions
# ==========================================================

prediction_df = pd.DataFrame(
    {
        "Prediction": predictions
    }
)

prediction_df.to_csv(
    "stm32_int8_predictions.csv",
    index=False
)

print("\nPredictions saved successfully.")

print("Output File : stm32_int8_predictions.csv")

print("\nValidation Finished.")
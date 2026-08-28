import serial
import csv
import os

# ==========================
# User Inputs
# ==========================

port = input("Enter COM Port (e.g., COM5): ")
label = input("Enter Activity Label (Idle/Pitch/Roll/Yaw/Shake): ")
num_samples = int(input("Enter Number of Samples to Record: "))

baudrate = 115200
filename = "dataset.csv"

# ==========================
# Open Serial Port
# ==========================

ser = serial.Serial(port, baudrate, timeout=1)

# ==========================
# Create CSV if Needed
# ==========================

file_exists = os.path.isfile(filename)

with open(filename, "a", newline="") as csvfile:

    writer = csv.writer(csvfile)

    if not file_exists:
        writer.writerow([
            "Ax",
            "Ay",
            "Az",
            "Gx",
            "Gy",
            "Gz",
            "Label"
        ])

    print("\nRecording Started...\n")

    count = 0

    while count < num_samples:

        line = ser.readline().decode().strip()

        values = line.split(",")

        if len(values) == 6:

            values.append(label)

            writer.writerow(values)

            count += 1

            print(f"Sample {count}/{num_samples}")

print("\nRecording Completed.")

ser.close()
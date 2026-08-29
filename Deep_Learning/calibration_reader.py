from pathlib import Path
import numpy as np

from onnxruntime.quantization import CalibrationDataReader


class MLPDataReader(CalibrationDataReader):
    """
    Calibration Data Reader for ONNX Runtime Static Quantization
    """

    def __init__(self):

        # --------------------------------------------------
        # Project Paths
        # --------------------------------------------------

        PROJECT_ROOT = Path(__file__).resolve().parent.parent

        CALIBRATION_DIR = PROJECT_ROOT / "Calibration"

        CALIBRATION_FILE = CALIBRATION_DIR / "calibration_data.npy"

        # --------------------------------------------------
        # Load Calibration Dataset
        # --------------------------------------------------

        self.data = np.load(CALIBRATION_FILE).astype(np.float32)

        self.current_index = 0

        print("=" * 60)
        print("Calibration Data Reader")
        print("=" * 60)

        print(f"Calibration Samples : {len(self.data)}")
        print(f"Feature Dimension   : {self.data.shape[1]}")

    # --------------------------------------------------
    # Return One Sample at a Time
    # --------------------------------------------------

    def get_next(self):

        if self.current_index >= len(self.data):
            return None

        sample = self.data[self.current_index]

        self.current_index += 1

        return {
            "input": sample.reshape(1, -1)
        }

    # --------------------------------------------------
    # Reset Reader
    # --------------------------------------------------

    def rewind(self):

        self.current_index = 0
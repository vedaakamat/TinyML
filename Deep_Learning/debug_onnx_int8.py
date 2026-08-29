import os
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "ONNX",
    "mlp_model_int8.onnx"
)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("\nINPUT DETAILS")
print("----------------------------")

for inp in session.get_inputs():
    print("Name :", inp.name)
    print("Shape:", inp.shape)
    print("Type :", inp.type)

print("\nOUTPUT DETAILS")
print("----------------------------")

for out in session.get_outputs():
    print("Name :", out.name)
    print("Shape:", out.shape)
    print("Type :", out.type)
import onnx

model = onnx.load("../ONNX/mlp_model_int8.onnx")

print("IR Version :", model.ir_version)

print("Opsets:")
for opset in model.opset_import:
    print(opset.domain, opset.version)
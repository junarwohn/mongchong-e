import tvm
from tvm import relay
import numpy as np
from tvm.contrib import utils, ndk
import onnx
import tvm.contrib.graph_executor as runtime

onnx_model = onnx.load("resnet152.onnx")

target = tvm.target.Target("llvm -mtriple=armv7l-linux-gnueabihf")

input_name = "input"  # Replace with actual input name if different
input_shape = (1, 3, 224, 224)  # Typical input shape for ResNet

mod, params = relay.frontend.from_onnx(onnx_model, shape={input_name: input_shape})

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

temp_dir = utils.tempdir()
lib_path = temp_dir.relpath("deploy_lib.tar")
lib.export_library(lib_path)




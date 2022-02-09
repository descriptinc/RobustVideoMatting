import torch_tensorrt
import torch

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").eval().cuda() # or "resnet50" 

traced_model = torch.jit.trace(model, [torch.ones((1, 12, 3, 1080, 1920)).to("cuda")])
trt_ts_module = torch_tensorrt.compile(traced_model,
    inputs = [
        torch_tensorrt.Input((1, 12, 3, 1080, 1920), dtype=torch.float32),
        torch_tensorrt.Input((1, 16, 144, 256), dtype=torch.float32),
        torch_tensorrt.Input((1, 16, 144, 256), dtype=torch.float32),
        torch_tensorrt.Input((1, 16, 144, 256), dtype=torch.float32),
        torch_tensorrt.Input((1, 16, 144, 256), dtype=torch.float32),
    ],
    enabled_precisions = {torch.float32},) # Run with FP16
#input_data = [torch.random([1, 12, 3, 1080, 1920]), torch.random([1, 16, 144, 256]), torch.random([1, 16, 144, 256]), torch.random([1, 16, 144, 256]), torch.random([1, 16, 144, 256])]
#result = trt_ts_module(input_data) # run inference
torch.jit.save(trt_ts_module, "RVM_trt_torchscript_module.ts") # save the TRT embedded Torchscript

from pathlib import Path

import argbind
import torch
from torch import nn
from torch.nn.utils import remove_weight_norm

from model import MattingNetwork

DEFAULT_INPUT_SIZE = [(1, 12, 3, 288, 512), (1, 16, 144, 256), (1, 20, 72, 128), (1, 40, 36, 64), (1, 64, 18, 32)]


def remove_weight_norm_layers(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d):
            try:
                remove_weight_norm(layer)
            except ValueError:
                pass


def get_onnx_session(model_file, check_conversion=False):
    import onnxruntime

    if check_conversion:
        import onnx

        # check model conversion
        onnx_model = onnx.load(model_file)
        onnx.checker.check_model(onnx_model)

    # load into session
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 0
    so.log_verbosity_level = 0
    ort_session = onnxruntime.InferenceSession(
        model_file, sess_options=so, providers=["CUDAExecutionProvider"]
    )

    return ort_session

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)

def to_onnx(model, save_path, inp_size=DEFAULT_INPUT_SIZE, opset=13):
    """
    Converts a given pytorch model object to onnx model
    """
    # remove_weight_norm_layers(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: the model will be optimized for a fixed SR with python int
    # change it to tensor in order for it to considered as a variable.
    inp = (
        torch.randn(inp_size[0], dtype=torch.float, device=device, requires_grad=False),
        torch.randn(inp_size[1], dtype=torch.float, device=device, requires_grad=False),
        torch.randn(inp_size[2], dtype=torch.float, device=device, requires_grad=False),
        torch.randn(inp_size[3], dtype=torch.float, device=device, requires_grad=False),
        torch.randn(inp_size[4], dtype=torch.float, device=device, requires_grad=False),
        # auto_downsample_ratio(*inp_size[-2:])
    )
    model.eval()
    onnx_path = save_path.with_suffix(".onnx")

    with torch.no_grad():
        # convert model
        torch.onnx.export(
            model,
            inp,
            str(onnx_path),
            opset_version=opset,
            export_params=True,
            input_names=[
                "video_data",
                "frame1_hidden_data",
                "frame2_hidden_data",
                "frame3_hidden_data",
                "frame4_hidden_data"
            ],  # the model's input names
            output_names=[
                "output_fgr",
                "output_alpha",
                "output_hidden_data_1",
                "output_hidden_data_2",
                "output_hidden_data_3",
                "output_hidden_data_4",
            ],  # the model's output names
            dynamic_axes={
                "video_data": {
                    0: "batch_size",
                },  # variable length axes
                "output_fgr": {0: "batch_size"},
                "output_alpha": {0: "batch_size"},
                "output_hidden_data_1": {0: "batch_size"},
                "output_hidden_data_2": {0: "batch_size"},
                "output_hidden_data_3": {0: "batch_size"},
                "output_hidden_data_4": {0: "batch_size"},
            },
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            keep_initializers_as_inputs=False,
            verbose=True,
        )


def to_tensorrt(model, save_path: Path, inp_size=DEFAULT_INPUT_SIZE):
    import onnx
    import tensorrt as trt
    from onnxsim import simplify

    # convert to onnx
    print("Step 1/3: Converting to onnx model")
    to_onnx(model, save_path=save_path, inp_size=inp_size)

    # simplify
    print("Step 2/3: Simplifying onnx model")
    onnx_path = save_path.with_suffix(".onnx")
    onnx_model = onnx.load(onnx_path)
    onnx_model, check = simplify(
        onnx_model, dynamic_input_shape=True, input_shapes={
            "video_data": inp_size[0],
            "frame1_hidden_data": inp_size[1],
            "frame2_hidden_data": inp_size[2],
            "frame3_hidden_data": inp_size[3],
            "frame4_hidden_data": inp_size[4],
            # "downsample_ratio": auto_downsample_ratio(inp_size[2:])
            }
    )
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, onnx_path)

    # convert to tensorrt engine
    print("Step 3/3: Converting onnx model to tensorrt")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(str(onnx_path))
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    assert success, "Unable to parse simplified onnx model"

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1024 MB
    config.flags = (1 << int(trt.BuilderFlag.FP16)) | (1 << int(trt.BuilderFlag.DEBUG))

    profile = builder.create_optimization_profile()
    profile.set_shape("video_data", DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[0])
    profile.set_shape("frame1_hidden_data", DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[1])
    profile.set_shape("frame2_hidden_data", DEFAULT_INPUT_SIZE[2], DEFAULT_INPUT_SIZE[2], DEFAULT_INPUT_SIZE[2])
    profile.set_shape("frame3_hidden_data", DEFAULT_INPUT_SIZE[3], DEFAULT_INPUT_SIZE[3], DEFAULT_INPUT_SIZE[3])
    profile.set_shape("frame4_hidden_data", DEFAULT_INPUT_SIZE[4], DEFAULT_INPUT_SIZE[4], DEFAULT_INPUT_SIZE[4])

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    trt_path = save_path.with_suffix(".trt")
    with open(trt_path, "wb") as fob:
        fob.write(serialized_engine)

    print("Finished tensorrt conversion!")


def get_tensorrt_engine(engine_path):
    import tensorrt as trt

    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    logger = trt.Logger(trt.Logger.WARNING)

    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


def to_openvino(model, save_path: Path, inp_size=DEFAULT_INPUT_SIZE):
    import onnx
    from onnxsim import simplify
    import shlex
    import subprocess

    # convert to onnx with opset 10 as it's the only one compatible :/
    print("Step 1/3: Converting to onnx model")
    to_onnx(model, save_path=save_path, inp_size=inp_size, opset=10)

    # simplify
    print("Step 2/3: Simplifying onnx model")
    onnx_path = save_path.with_suffix(".onnx")
    onnx_model = onnx.load(onnx_path)
    onnx_model, check = simplify(
        onnx_model, dynamic_input_shape=True, input_shapes={"audio_data": inp_size}
    )
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, onnx_path)

    # convert to openvino engine
    print("Step 3/3: Converting onnx model to OpenVino")

    # disable fusing as it produces error for now
    command = (
        f"mo --input_model {onnx_path} "
        + f"--output_dir {onnx_path.parent} "
        + f"--framework onnx --static_shape --data_type=FP16 "
        + f"--input_shape=[{','.join(inp_size)}] --disable_fusing"
    )
    command = shlex.split(command)

    subprocess.check_output(command)


def get_openvino_engine(xml_path, bin_path):
    import multiprocessing
    from openvino.inference_engine import IECore

    ie_core_handler = IECore()
    network = ie_core_handler.read_network(model=xml_path, weights=bin_path)
    executable_network = ie_core_handler.load_network(
        network,
        device_name="CPU",
        num_requests=1,
        config={"CPU_THREADS_NUM": f"{multiprocessing.cpu_count() - 1}"},
    )
    return executable_network


CONVERTER_MAP = {"tensorrt": to_tensorrt, "openvino": to_openvino, "onnx": to_onnx}


def convert(model_path: Path, model_type: str, input_size: tuple = DEFAULT_INPUT_SIZE):
    """
    Converts the given pytorch model and saves it in the parent directory with
    relevant extension. For tensorrt, extension is .trt. For openvino, two
    files are saved with .bin and .xml extensions. For onnx, extension is .onnx.
    Existing files will be silently overwritten.
    Parameters
    ----------
    model_path : str
        .pth file path that needs to be converted
    model_type: str
        One of "tensorrt", "openvino", "onnx"
    input_size: tuple
        default shape is (1,1,5*44100)
    """
    converter_fn = CONVERTER_MAP.get(model_type, None)
    if converter_fn is None:
        print(f"No converters found for model type {model_type}")

    model = MattingNetwork(variant='mobilenetv3').eval().to('cuda') # Or variant="resnet50"
    model.load_state_dict(torch.load(model_path))
    converter_fn(model, save_path=model_path, inp_size=input_size)


if __name__ == "__main__":
    convert = argbind.bind(convert, without_prefix=True, positional=True)
    args = argbind.parse_args()
    with argbind.scope(args):
        convert()

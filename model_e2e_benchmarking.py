"""
Benchmarking script for models
"""
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import wav2wav
from tests.memory_profiler import memory_usage
from wav2wav.interface import ModelType

SR = 44100
TEST_DURATIONS = [30, 60, 60 * 6]  # duration in minutes
TEST_CONFIGS = ["slow"]
ASSET_DIR = Path(__file__).resolve().parent / "assets"


@dataclass
class ProfileStat:
    latency_mean: float = 0.0  # latency in seconds
    latency_std: float = 0.0
    memory_mean: float = 0.0  # memory in mb
    memory_std: float = 0.0
    gpu_memory_mean: float = 0.0  # memory in mb
    gpu_memory_std: float = 0.0
    runs: int = 0

    # list of stats to compare.
    # first elem is the attr name.
    # second is cmp operator, read "lower"/"higher" is better.
    # third is the std deviation stat to use when strict comparison fails, write "null" for strict comparison
    # Don't provide type-hint. This is a hack to avoid it being counted as a dataclass Field.
    cmp_stats = [
        ("latency_mean", "lower", "latency_std"),
        ("memory_mean", "lower", "memory_std"),
        ("gpu_memory_mean", "lower", "gpu_memory_std"),
    ]


def profile(
    fn: Callable, fn_args: List = [], fn_kwargs: Dict = {}, runs: int = 10
) -> Dict[str, Union[ProfileStat, Tensor, AudioSignal]]:
    """
    Profiles a given function. Stats are computed over number of ``runs``.

    NOTE: GPU memory is profiled based on the ``fn_kwargs['device']`` object.
    Thus, this object must be specified else a ``KeyError`` will be raised.
    GPU memory is profiled for the whole GPU instead of a given process. Refer
    memory_profiler._get_gpu_memroy for more information.

    NOTE: Memory profiling is done by periodically gathering stats. Only the
    maximum memory used is collected and mean of these maximums is calculated over
    the specified number of runs.

    Args:
        fn: Function to be profiled.
        fn_args: Function args.
        fn_kwargs: Function kwargs.
        runs: Number of runs to compute stats over.

    Returns:
        Dictionary object as follows:
            - stats: ProfileStat object.
            - output: Output from fn
    """

    # compute end-to-end latency
    def _latency(*_fn_args, **_fn_kwargs):
        collect_out = _fn_kwargs.pop("_collect_out", False)
        t1 = time()
        out = fn(*_fn_args, **_fn_kwargs)
        t2 = time()
        res = t2 - t1
        if not collect_out:
            del out
            out = None
        return res, out

    # compute max memory consumption
    def _memory(_fn, _fn_args, _fn_kwargs) -> float:
        gpu_index = (
            _fn_kwargs["device"].index if _fn_kwargs["device"].type == "cuda" else None
        )
        (max_mem, max_gpu_mem), fn_res = memory_usage(
            (_fn, _fn_args, _fn_kwargs),
            interval=0.01,
            retval=True,
            max_usage=True,
            gpu_index=gpu_index,
        )
        return max_mem, max_gpu_mem, fn_res

    stats = ProfileStat()
    mem_prof_res = []

    # combine memory and e-e latency computation
    for i in tqdm(range(runs)):
        # no need to collect out from all runs.
        # this is to prevent memory overflow during stress testing.
        fn_kwargs["_collect_out"] = False
        if i == 0:
            fn_kwargs["_collect_out"] = True
        mem_prof_res.append(_memory(_latency, fn_args, fn_kwargs))

    mem_usage = [x[0] for x in mem_prof_res]
    gpu_mem_usage = [x[1] for x in mem_prof_res]
    latencies = [x[2][0] for x in mem_prof_res]

    # pick any run, choose the output of _latency, extract the output
    output = mem_prof_res[0][2][1]

    stats.latency_mean, stats.latency_std = np.mean(latencies), np.std(latencies)
    stats.memory_mean, stats.memory_std = np.mean(mem_usage), np.std(mem_usage)
    stats.gpu_memory_mean, stats.gpu_memory_std = np.mean(gpu_mem_usage), np.std(
        gpu_mem_usage
    )
    stats.runs = runs
    return {"stats": stats, "output": output}


def load(duration: int) -> AudioSignal:
    """
    Load input audio of specified duration.

    Args:
        duration: Length of audio in seconds.

    Returns:
        AudioSignal of specified duration.
    """
    # reseed at every invocation for consistency across multiple tests
    wav2wav.funcs.seed(42)
    long_audio = AudioSignal(
        torch.tanh(torch.randn(int(SR * duration), dtype=torch.float32)), SR
    )

    if int(duration) == 6:  # 1/10*60 minutes:
        asset_file = ASSET_DIR / "cpu_test_file.pt"
        if asset_file.exists():
            long_audio = torch.load(open(asset_file, "rb"))
            long_audio = AudioSignal(long_audio)
        else:
            torch.save(long_audio.audio_data, asset_file)

    return long_audio


def enhance(**kwargs):
    # important to seed between invocation otherwise output produced is not
    # byte equal for the same input. This has to do with bias computation step.
    # This step uses a normally distributed tensor. In the absence of seeding,
    # this tensor changes and we get different output for the same input.
    wav2wav.funcs.seed(42, set_cudnn=kwargs.pop("set_cudnn", False))
    return wav2wav.interface.enhance(**kwargs)


def handle_model_conversion(config: str, model_type: ModelType, model_path: Path):
    """
    If model files are absent, converts an existing pytorch model to given model type.
    If pytorch model is absent, recreate the model and save weights at given location.
    """
    if model_type in [None, ModelType.PYTORCH] and not model_path.exists():
        # reseed at every invocation for consistency across multiple tests
        wav2wav.funcs.seed(42)
        model = wav2wav.modules.Generator(config=config, sample_rate=SR)
        model.save(model_path, package=False)
    elif model_type == ModelType.ONNX and not model_path.with_suffix(".onnx").exists():
        print("Converting to ONNX")
        wav2wav.converter.convert(model_path, "onnx")
    elif model_type == ModelType.OPENVINO and not (
        model_path.with_suffix(".bin").exists()
        and model_path.with_suffix(".xml").exists()
    ):
        print("Converting to OpenVino")
        wav2wav.converter.convert(model_path, "openvino")
    elif (
        model_type == ModelType.TENSORRT and not model_path.with_suffix(".trt").exists()
    ):
        print("Converting to TensorRT")
        wav2wav.converter.convert(model_path, "tensorrt")


def get_model_path(config: str, model_type: ModelType = None) -> Path:
    """
    Returns saved model path for a given config. If the saved model doesn't exists,
    creates a new random saved model.

    Args:
        config: Model config type. Must be one of ``wav2wav.modules.Generator.CONFIGS``.

    Returns:
        Saved model path.
    """
    model_path = ASSET_DIR / f"{config}.pth"
    handle_model_conversion(config, model_type, model_path)
    return model_path


def main(runs=5, durations=TEST_DURATIONS, configs=TEST_CONFIGS, enhance_fn_kwargs={}):
    res = {}
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    for duration in durations:
        x = load(duration * 60)
        for config in configs:
            model_path = get_model_path(
                config, enhance_fn_kwargs.get("model_type", None)
            )
            enhance_fn_kwargs.update(
                {
                    "audio_path_or_signal": x,
                    "model_path": model_path,
                    "device": device,
                    "no_match_input_db": True,
                    "normalize_db": -24,
                }
            )
            prof_out = profile(
                enhance,
                runs=runs,
                fn_kwargs=enhance_fn_kwargs,
            )
            # place input for evaluation
            prof_out["input"] = x
            res[(duration, config)] = prof_out
    return res


if __name__ == "__main__":
    print(
        "WARNING: RUN THIS SCRIPT IN COMPLETE ISOLATION! KILL ALL OTHER PROCESSES ON CUDA:0"
    )
    enhance_fn_kwargs = {"model_type": wav2wav.interface.ModelType.TENSORRT}
    res = main(runs=3, enhance_fn_kwargs=enhance_fn_kwargs)
    for k, v in res.items():
        print(f"Duration: {k[0]} | Config: {k[1]} | Stats: {v['stats']}")

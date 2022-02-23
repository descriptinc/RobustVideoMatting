"""
Benchmarking script for models
"""
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from time import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
import pprint
import pandas as pd

import argparse
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tqdm import tqdm

from memory_profiler import memory_usage
from model import MattingNetwork
from inference import convert_video

TEST_DURATIONS = ['10s', '1m', '5m']
TEST_FPS = [24, 30, 60]
# TEST_RESOLUTIONS = [240, 360, 480, 720]
TEST_RESOLUTIONS = [720]
TEST_DOWNSAMPLING_RATIOS = [0.125, 0.25, 0.5]
TEST_PRECISIONS = ['float16', 'float32']
# TEST_CONFIGS = ["slow"]

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
) -> Dict[str, Union[ProfileStat, Tensor]]:
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


def get_asset_file_name(asset_dir, fps: int, resolution:int, duration: str):
    """
    Load input audio of specified duration.

    Args:
        duration: Length of audio in seconds.

    Returns:
        source filename
    """
    # TODO: take care of seeding
    asset_file = Path(asset_dir) / f"test_fps{fps}_res{resolution}_t{duration}.mp4"
    return asset_file

def main(model_file, asset_dir, runs=5):
    result = []
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = MattingNetwork("mobilenetv3").cuda()
    model.load_state_dict(torch.load(model_file, map_location=device))
    for duration in durations:
        for fps in TEST_FPS:
            for res in TEST_RESOLUTIONS:
                for ratio in TEST_DOWNSAMPLING_RATIO:
                    for precision in TEST_PRECISIONS:
                        model = model.to(dtype=precision)
                        input_asset_file_name = get_asset_file_name(asset_dir, fps, res, duration)
                        if input_asset_file_name.exists():
                            input_asset_file_name = str(input_asset_file_name)
                        else:
                            continue
                        output_asset_file_name = input_asset_file_name.split('.mp4')[0] + '_alpha.mp4'
                        fn_kwargs = {
                            'model': model,
                            'seq_chunk': 14,
                            'input_source': input_asset_file_name,
                            '_collect_out': False,
                            'device': device,
                            'output_alpha': output_asset_file_name,
                            'downsample_ratio': downsample_ratio
                        }
                        prof_out = profile(
                            convert_video,
                            runs=runs,
                            fn_kwargs=fn_kwargs,
                        )
                        # place input for evaluation
                        prof_out["input"] = input_asset_file_name
                        prof_out["fps"] = fps
                        prof_out["resolution"] = res
                        prof_out["duration"] = duration
                        prof_out.update(asdict(prof_out["stats"]))
                        prof_out.pop("stats", None)
                        result.append(prof_out)
    return result


if __name__ == "__main__":
    print(
        "WARNING: RUN THIS SCRIPT IN COMPLETE ISOLATION! KILL ALL OTHER PROCESSES ON CUDA:0"
    )
    argparser = argparse.ArgumentParser(description='inputs for benchmarking')
    argparser.add_argument("model", type=str, help='model used for benchmarking')
    argparser.add_argument("asset_dir", type=str, help='location that contains all test input videos')
    argparser.add_argument("stats_output_file", type=str, help='file that contains the benchmark outputs')
    args = argparser.parse_args()
    result = main(args.model, args.asset_dir, runs=3)
    result_df = pd.DataFrame(data=result).round(2)
    result_df.to_csv(args.stats_output_file)
    print(result_df)

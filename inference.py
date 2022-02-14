"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Callable
from tqdm.auto import tqdm
from model_converter import get_tensorrt_engine, DEFAULT_INPUT_SIZE

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter

def convert_video(inference_fn: Callable,
                  input_source: str,
                  device: str,
                  dtype: torch.dtype,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    
    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')
    # if device is None or dtype is None:
    #     param = next(model.parameters())
    #     dtype = param.dtype
    #     device = param.device

    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
        # rec = [None] * 4
        rec = [
            torch.zeros(DEFAULT_INPUT_SIZE[1]).to(device), 
            torch.zeros(DEFAULT_INPUT_SIZE[2]).to(device), 
            torch.zeros(DEFAULT_INPUT_SIZE[3]).to(device), 
            torch.zeros(DEFAULT_INPUT_SIZE[4]).to(device)]
        for src in reader:

            if downsample_ratio is None:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
            if downsample_ratio != 1:
                src_sm = interpolate(src, scale_factor=downsample_ratio)
            import pdb; pdb.set_trace()
            # forward pass
            src_sm = src_sm.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]

            fgr, pha, *rec = inference_fn(src_sm, *rec)

            if output_foreground is not None:
                writer_fgr.write(fgr[0])
            if output_alpha is not None:
                writer_pha.write(pha[0])
            if output_composition is not None:
                if output_type == 'video':
                    com = fgr * pha + bgr * (1 - pha)
                else:
                    fgr = fgr * pha.gt(0)
                    com = torch.cat([fgr, pha], dim=-3)
                writer_com.write(com[0])

            bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def interpolate(x: Tensor, scale_factor: float):
    if x.ndim == 5:
        B, T = x.shape[:2]
        x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
            mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x = x.view([B, T] + list(x.shape[1:]))
    else:
        x = F.interpolate(x, scale_factor=scale_factor,
            mode='bilinear', align_corners=False, recompute_scale_factor=False)
    return x


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def pytorch_inference_fn(device, model_path):
    model = MattingNetwork('mobilenetv3').eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = torch.jit.script(model)
    model = torch.jit.freeze(model)
    def inference_fn(video_data, frame1_hidden_data, frame2_hidden_data, frame3_hidden_data, frame4_hidden_data):
        with torch.no_grad():
            fgr, pha, *rec = model(video_data, frame1_hidden_data, frame2_hidden_data, frame3_hidden_data, frame4_hidden_data)
        return fgr, pha, *rec
    return inference_fn


def tensorrt_inference_fn(device, model_path):
    from model_converter import get_tensorrt_engine, DEFAULT_INPUT_SIZE

    model_path = Path(model_path)
    engine_path = model_path.with_suffix(".trt")
    import pdb; pdb.set_trace()
    if not engine_path.exists():
        raise FileNotFoundError(f"Unable to find tensorrt engine at {engine_path}")

    engine = get_tensorrt_engine(str(engine_path))
    context = engine.create_execution_context()

    # TODO: allow variable sample rate as input. the trace freezes the sample_rate
    # value now.
    input_names = ["video_data", "frame1_hidden_data", "frame2_hidden_data", "frame3_hidden_data", "frame4_hidden_data"]
    output_names = ["output_fgr", "output_alpha", "output_hidden_data_1", "output_hidden_data_2", "output_hidden_data_3", "output_hidden_data_4"]
    input_indices = [engine[input_names[i]] for i in range(len(input_names))]
    output_indices = [engine[output_names[i]] for i in range(len(output_names))]

    def inference_fn(video_data, frame1_hidden_data, frame2_hidden_data, frame3_hidden_data, frame4_hidden_data, device=device):
        output_fgr = torch.zeros_like(video_data)
        output_alpha = torch.zeros_like(video_data)
        output_hidden_data_1 = torch.zeros(DEFAULT_INPUT_SIZE[1])
        output_hidden_data_2 = torch.zeros(DEFAULT_INPUT_SIZE[2])
        output_hidden_data_3 = torch.zeros(DEFAULT_INPUT_SIZE[3])
        output_hidden_data_4 = torch.zeros(DEFAULT_INPUT_SIZE[4])
        buffers = [None] * 11
        buffers[input_indices[0]] = video_data.data_ptr()
        # if frame1_hidden_data is not None:
        buffers[input_indices[1]] = frame1_hidden_data.data_ptr() if frame1_hidden_data is not None else torch.zeros(DEFAULT_INPUT_SIZE[1])
        # if frame2_hidden_data is not None:
        buffers[input_indices[2]] = frame2_hidden_data.data_ptr() if frame2_hidden_data is not None else torch.zeros(DEFAULT_INPUT_SIZE[2])
        # if frame3_hidden_data is not None:
        buffers[input_indices[3]] = frame3_hidden_data.data_ptr() if frame3_hidden_data is not None else torch.zeros(DEFAULT_INPUT_SIZE[3])
        # if frame4_hidden_data is not None:
        buffers[input_indices[4]] = frame4_hidden_data.data_ptr() if frame4_hidden_data is not None else torch.zeros(DEFAULT_INPUT_SIZE[4])
        buffers[output_indices[0]] = output_fgr.data_ptr()
        buffers[output_indices[1]] = output_alpha.data_ptr()
        buffers[output_indices[2]] = output_hidden_data_1.data_ptr()
        buffers[output_indices[3]] = output_hidden_data_2.data_ptr()
        buffers[output_indices[4]] = output_hidden_data_3.data_ptr()
        buffers[output_indices[5]] = output_hidden_data_4.data_ptr()
        status = context.execute_v2(buffers)
        if status is not True:
            raise RuntimeError("TensorRT failed to run")
        return output_fgr, output_alpha, output_hidden_data_1, output_hidden_data_2, output_hidden_data_3, output_hidden_data_4

    return inference_fn

class Converter:
    def __init__(self, inference_fn: Callable, device: str):
        self.inference_fn = inference_fn
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.inference_fn, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
    import argparse
    from model import MattingNetwork
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()
    # if the model is in the tensorrt version
    inference_fn = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if '.trt' in args.checkpoint:
        inference_fn = tensorrt_inference_fn(device, args.checkpoint)
    else:
        inference_fn = pytorch_inference_fn(device, args.checkpoint)

    converter = Converter(inference_fn, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )

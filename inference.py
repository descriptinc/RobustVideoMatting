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

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Tuple
from tqdm.auto import tqdm

import math

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter

torch.backends.cudnn.benchmark=True

def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  batch_size: int = 1,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    
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
    assert batch_size >= 1, 'Batch size must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    time_prof = {}
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    import time
    # start time
    start_time_t0 = time.time()
    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    data_set_init_time_t1 = time.time()
    device_count = torch.cuda.device_count()
    def chunking_collate_fn(batch):
        batch = torch.stack(batch, dim=0)
        curr_batch_size = batch.shape[0]
        if curr_batch_size % seq_chunk == 0:
            return batch.view([-1, seq_chunk] + list(batch.shape)[1:])
        else:
            def maxPrimeFactor(n):
                max_Prime = n
                # number must be even
                while n % 2 == 0:
                    max_Prime = 2
                    n /= 2
                # number must be odd
                for i in range(3, int(math.sqrt(n)) + 1, 2):
                    while n % i == 0:
                        max_Prime = i
                        n = n / i
                # prime number greator than two
                if n > 2:
                    max_Prime = n
                return int(max_Prime)
            seq_length = max(maxPrimeFactor(curr_batch_size), curr_batch_size)
            return batch.view([-1, seq_length] + list(batch.shape)[1:])

    reader = DataLoader(source, batch_size=batch_size * seq_chunk, pin_memory=True, num_workers=num_workers, collate_fn=chunking_collate_fn)
    data_loader_init_time_t2 = time.time()

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

    # Inference
    model = model.eval().to(device)
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    # model = torch.nn.DataParallel(model, device_ids=[0,1])
    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        # with torch.no_grad():
        with torch.inference_mode():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            data_loader_read_tot_time = 0
            total_runs = 0
            forward_pass_tot_time = 0
            data_loader_read_start_time_t3 = time.time()
            for_loop_start_time = time.time()
            output_writing_tot_time = 0
            for src in reader:
                data_loader_read_end_time_t4 = time.time()
                data_loader_read_tot_time += data_loader_read_end_time_t4 - data_loader_read_start_time_t3
                total_runs += 1
                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[3:])

                # src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                # batch_size = max(1, src.shape[0]//seq_chunk)
                # seq_length = min(src.shape[0], seq_chunk)
                # src = src.view([batch_size, seq_length] + list(src.shape)[1:]).to(device, dtype, non_blocking=True) # [B, T, C, H, W]
                src = src.to(device, dtype, non_blocking=True)
                if batch_size != src.shape[0]:
                    for idx,_ in enumerate(rec):
                        rec[idx] = rec[idx][:src.shape[0]]
                model_fwd_pass_start_time_t5 = time.time()
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                print("doing forward pass")
                fgr, pha, *rec = model(src, *rec, downsample_ratio)
                model_fwd_pass_start_time_t6 = time.time()
                forward_pass_tot_time += model_fwd_pass_start_time_t6 - model_fwd_pass_start_time_t5
                fgr = fgr.view([1, -1] + list(fgr.shape)[2:])
                pha = pha.view([1, -1] + list(pha.shape)[2:])

                output_writing_start_time_t7 = time.time()
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
                output_writing_end_time_t8 = time.time()
                output_writing_tot_time += output_writing_end_time_t8 - output_writing_start_time_t7
                bar.update(src.size(1))
                data_loader_read_start_time_t3 = time.time()
            for_loop_end_time = time.time()

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()
        # end time
        end_time = time.time()

        time_prof['data_read_total_time'] = data_loader_read_tot_time
        time_prof['data_read_time_per_batch'] = data_loader_read_tot_time/total_runs
        time_prof['forward_pass_total_time'] = forward_pass_tot_time
        time_prof['forward_pass_per_batch'] = forward_pass_tot_time/total_runs
        time_prof['output_writing_total_time'] = output_writing_tot_time
        time_prof['output_writing_per_batch'] = output_writing_tot_time/total_runs
        time_prof['total_runs'] = total_runs
        time_prof['total_time'] = end_time-start_time_t0
        return time_prof
        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="cuda_time_total"))


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        # self.model = torch.jit.script(self.model)
        # self.model = torch.jit.freeze(self.model)
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
    import argparse
    from model import MattingNetwork
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
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
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()
    
    converter = Converter(args.variant, args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        batch_size=args.batch_size,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )
    

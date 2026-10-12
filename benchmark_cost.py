"""Measures the inference cost reported in the paper (Table 10).

For TabCNN (as described by Wiggins and Kim, 2019), the single-task HCQT+Mel
model, its parameter-matched variant and the proposed cascaded model, reports
trainable parameters, multiply-accumulates per output frame (Conv2d and Linear
layers only; bias, normalization and activations excluded) and the real-time
factor of the network alone (feature extraction excluded), timed in float32 on
a batch of 431 frames (10 s of audio): median of 30 runs after 10 warm-ups on
GPU, median of 5 runs after 2 warm-ups on CPU.

Example:
    python benchmark_cost.py --output cost.json
    python benchmark_cost.py --no-timing      # parameters and MACs only
"""
import argparse
import json
import platform
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models import get_model
from src.utils.config_helpers import process_config

MODELS = [
    ("Single-task HCQT+Mel", "configs/ablations/hcqt_mel_single.yaml"),
    ("Single-task HCQT+Mel, parameter-matched", "configs/ablations/B_param_matched.yaml"),
    ("Cascaded MTL HCQT+Mel", "configs/cnn_mtl_config.yaml"),
]


def tabcnn_reference() -> nn.Module:
    """TabCNN as described by Wiggins and Kim (2019): 192-bin CQT, 9-frame context."""
    class TabCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3), nn.ReLU(),
                nn.Conv2d(32, 64, 3), nn.ReLU(),
                nn.Conv2d(64, 64, 3), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25), nn.Flatten(),
                nn.Linear(64 * 93, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 126),
            )

        def forward(self, x):
            return {"tab_logits": self.net(x["cqt192"])}

    return TabCNN()


def dummy_inputs(config: dict, n_frames: int, device) -> dict:
    """Random context windows shaped like the model's inputs, one per output frame."""
    window = config["data"].get("framify_window_size", 19)
    defs = config["feature_definitions"]
    return {k: torch.randn(n_frames, defs[k]["in_channels"], defs[k]["num_freq"], window, device=device)
            for k in config["data"]["active_features"]}


def count_macs(model: nn.Module, inputs: dict) -> int:
    """Multiply-accumulates of Conv2d and Linear layers for one output frame (inputs hold one frame)."""
    total = [0]

    def conv_hook(m, _, out):
        total[0] += out.numel() // out.shape[0] * m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)

    def linear_hook(m, _, out):
        total[0] += out.numel() // out.shape[0] * m.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(inputs)
    for h in hooks:
        h.remove()
    return total[0]


def time_forward(model: nn.Module, make_input, device, n_frames: int, warmup: int, repeats: int) -> tuple[float, float]:
    """Median and standard deviation, in seconds, of one forward pass over `n_frames` frames."""
    x = make_input(n_frames, device)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return process_config(yaml.safe_load(f))


def measure_cost(timing: bool = True, seconds: float = 10.0) -> dict:
    """Runs the cost protocol on CPU, and on GPU too if one is available."""
    configs = {name: load_config(path) for name, path in MODELS}
    data_conf = next(iter(configs.values()))["data"]
    frames_per_second = data_conf["sample_rate"] / data_conf["hop_length"]
    n_frames = int(round(seconds * frames_per_second))
    gpu_warmup, gpu_repeats, cpu_warmup, cpu_repeats = 10, 30, 2, 5

    models = [("TabCNN (Wiggins and Kim, 2019)", tabcnn_reference,
               lambda n, dev: {"cqt192": torch.randn(n, 1, 192, 9, device=dev)})]
    models += [(name, (lambda c=cfg: get_model(c)), (lambda n, dev, c=cfg: dummy_inputs(c, n, dev)))
               for name, cfg in configs.items()]

    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    rows = []
    for name, make_model, make_input in models:
        torch.manual_seed(0)
        model = make_model()
        with torch.no_grad():
            model(make_input(2, cpu))  # materialize LazyLinear layers
        model.eval()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        macs = count_macs(model, make_input(1, cpu))
        row = {"model": name, "params": params, "params_M": params / 1e6,
               "macs_per_frame": macs, "macs_M_per_frame": macs / 1e6,
               "gmacs_per_second_audio": macs * frames_per_second / 1e9}
        if timing:
            if gpu is not None:
                model = model.to(gpu)
                median, std = time_forward(model, make_input, gpu, n_frames, gpu_warmup, gpu_repeats)
                row.update(gpu_ms_per_second_audio=1000 * median / seconds, gpu_ms_std=1000 * std / seconds,
                           gpu_rtf=median / seconds)
                model = model.to(cpu)
            median, std = time_forward(model, make_input, cpu, n_frames, cpu_warmup, cpu_repeats)
            row.update(cpu_ms_per_second_audio=1000 * median / seconds, cpu_ms_std=1000 * std / seconds,
                       cpu_rtf=median / seconds, cpu_threads=torch.get_num_threads())
        rows.append(row)
        print(f"  {name:42s} {params / 1e6:6.3f} M params  {macs / 1e6:8.1f} M MACs/frame"
              + (f"  GPU RTF {row['gpu_rtf']:.5f}" if "gpu_rtf" in row else "")
              + (f"  CPU RTF {row['cpu_rtf']:.3f}" if "cpu_rtf" in row else ""))
        del model

    return {
        "rows": rows,
        "protocol": {
            "batch": f"{n_frames} frames ({seconds:.0f} s of audio)",
            "gpu": torch.cuda.get_device_name(0) if gpu is not None else None,
            "gpu_warmup": gpu_warmup, "gpu_repeats": gpu_repeats,
            "cpu": platform.processor() or platform.machine(),
            "cpu_warmup": cpu_warmup, "cpu_repeats": cpu_repeats,
            "statistic": "median", "precision": "float32",
            "feature_extraction": "excluded",
            "macs": "Conv2d and Linear only; bias, normalization and activations excluded",
            "timing": timing,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=None, help="Write the results to this JSON file.")
    parser.add_argument("--no-timing", action="store_true", help="Only count parameters and MACs.")
    args = parser.parse_args()

    result = measure_cost(timing=not args.no_timing)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()

"""
FLOPs measurement utilities for BATG budget sweep experiments.

Two approaches used for cross-checking:
  1. fvcore  — Facebook's library, profile-based, handles dynamic models well
  2. ptflops — simpler, good for standard CNN/MLP layers

Install: pip install fvcore ptflops
Both already in requirements.txt.

For BATG specifically:
  - Measure FLOPs of the FULL pipeline at B=100% (all segments processed)
  - Measure FLOPs at B=25/50/75% (only kept segments go through backbone)
  - Report: flops_at_B / flops_at_100% × 100 = actual compute percentage used
  - Also report wall-clock FPS and latency separately (FLOPs ≠ always = speedup)
"""

import torch
import torch.nn as nn


def count_flops_ptflops(model: nn.Module,
                         input_shape: tuple,
                         device: str = "cpu") -> dict:
    """Count FLOPs using ptflops library.
    
    Args:
        model: PyTorch model
        input_shape: input tensor shape WITHOUT batch dim, e.g. (10, 1024) for one segment
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with 'flops' (float, in GFLOPs) and 'params' (int)
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        raise ImportError("Run: pip install ptflops")

    model = model.to(device)
    macs, params = get_model_complexity_info(
        model, input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    # ptflops returns MACs; multiply by 2 for FLOPs (each MAC = 1 multiply + 1 add)
    flops = macs * 2
    return {
        "flops_G": flops / 1e9,
        "params_M": params / 1e6,
    }


def count_flops_fvcore(model: nn.Module,
                        example_input: torch.Tensor) -> dict:
    """Count FLOPs using fvcore library.
    
    Args:
        model: PyTorch model
        example_input: a real input tensor (used for tracing)
    
    Returns:
        dict with 'flops_G' (GFLOPs) and 'params_M' (millions of params)
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
    except ImportError:
        raise ImportError("Run: pip install fvcore")

    flops = FlopCountAnalysis(model, example_input)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    total_flops = flops.total()
    total_params = sum(parameter_count(model).values())

    return {
        "flops_G": total_flops / 1e9,
        "params_M": total_params / 1e6,
    }


def measure_fps(model: nn.Module,
                example_input: torch.Tensor,
                n_runs: int = 50,
                device: str = "cpu") -> dict:
    """Measure inference FPS and latency by timing repeated forward passes.
    
    Args:
        model: PyTorch model in eval mode
        example_input: input tensor (one video's segments)
        n_runs: number of timed runs (more = more stable estimate)
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with 'fps', 'latency_ms_mean', 'latency_ms_std'
    """
    import time
    import numpy as np

    model = model.to(device).eval()
    example_input = example_input.to(device)

    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(example_input)

    # timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(example_input)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # convert to ms
    return {
        "fps": 1000.0 / np.mean(times),
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
    }


if __name__ == "__main__":
    # smoke test with a tiny dummy model
    dummy = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    x = torch.randn(32, 1024)  # 32 segments, 1024-dim features

    print("--- ptflops ---")
    try:
        r = count_flops_ptflops(dummy, input_shape=(1024,))
        print(f"  FLOPs: {r['flops_G']:.4f} G  |  Params: {r['params_M']:.4f} M")
    except ImportError as e:
        print(f"  Skipped: {e}")

    print("--- fvcore ---")
    try:
        r = count_flops_fvcore(dummy, x)
        print(f"  FLOPs: {r['flops_G']:.4f} G  |  Params: {r['params_M']:.4f} M")
    except ImportError as e:
        print(f"  Skipped: {e}")

    print("--- FPS / latency (CPU) ---")
    r = measure_fps(dummy, x, n_runs=20, device="cpu")
    print(f"  FPS: {r['fps']:.1f}  |  Latency: {r['latency_ms_mean']:.2f} ± {r['latency_ms_std']:.2f} ms")

    print("Smoke test passed.")
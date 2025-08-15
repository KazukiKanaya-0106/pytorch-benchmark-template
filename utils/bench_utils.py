import time
import torch
import torch.nn as nn
from typing import Callable, Any

from utils.torch_utils import TorchUtils


class BenchUtils:
    @staticmethod
    def count_params(model: nn.Module, trainable_only: bool = False) -> int:
        params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in params)

    @staticmethod
    @torch.inference_mode()
    def measure_inference_speed(
        model: nn.Module,
        forward_fn: Callable[[nn.Module, Any], Any],
        batch: Any,
        device: torch.device,
        warmup: int = 10,
        iters: int = 50,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ) -> dict:
        model.eval()
        X, _ = TorchUtils.split_batch(batch)
        X = TorchUtils.move_to_device(X, device)
        bs = BenchUtils.batch_size_from_X(X)

        if device.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()

            with autocast_ctx:
                for _ in range(warmup):
                    _ = forward_fn(model, X)
            torch.cuda.synchronize()

            times_ms = []
            with autocast_ctx:
                for _ in range(iters):
                    starter.record()
                    _ = forward_fn(model, X)
                    ender.record()
                    torch.cuda.synchronize()
                    times_ms.append(starter.elapsed_time(ender))  # ms
            avg_ms = float(sum(times_ms) / len(times_ms))
            sm = sorted(times_ms)
            p50_ms = float(sm[len(sm) // 2])
            p90_ms = float(sm[int(len(sm) * 0.9)])
            total_s = sum(times_ms) / 1000.0

        else:
            for _ in range(warmup):
                _ = forward_fn(model, X)

            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = forward_fn(model, X)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)  # ms
            avg_ms = float(sum(times) / len(times))
            sm = sorted(times)
            p50_ms = float(sm[len(sm) // 2])
            p90_ms = float(sm[int(len(sm) * 0.9)])
            total_s = sum(times) / 1000.0

        throughput = (bs * iters) / total_s
        return {
            "batch_size": float(bs),
            "latency_ms_avg": avg_ms,
            "latency_ms_p50": p50_ms,
            "latency_ms_p90": p90_ms,
            "latency_ms_per_sample": avg_ms / bs,
            "throughput_sps": float(throughput),
            "iters": float(iters),
            "warmup": float(warmup),
            "amp": float(bool(use_amp)),
            "amp_dtype": str(amp_dtype),
            "device": device.type,
        }

    @staticmethod
    def batch_size_from_X(X: Any) -> int:
        if isinstance(X, torch.Tensor):
            return X.size(0)
        if isinstance(X, (list, tuple)):
            return BenchUtils.batch_size_from_X(X[0])
        if isinstance(X, dict):
            return BenchUtils.batch_size_from_X(next(iter(X.values())))
        raise TypeError(f"Unsupported batch type: {type(X)}")

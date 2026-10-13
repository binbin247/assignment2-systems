"""Benchmark the basics Transformer for CS336's benchmarking_script problem.

Run from the assignment root (with cs336_basics installed):
    uv run python -m cs336_systems.benchmarking_script --model-size small --mode forward
    uv run python -m cs336_systems.benchmarking_script --mode forward-backward
    uv run python -m cs336_systems.benchmarking_script --mode train --timing breakdown

Defaults match the handout: FP32, batch 4, context 512, vocabulary 10,000,
5 warm-up steps and 10 measured steps. Use --warmup-steps 0, 1, or 2 in
separate processes for part (c), and --output to keep the individual timings.
The device defaults to CUDA when available, otherwise CPU.

End-to-end timing synchronizes only at step boundaries. Breakdown timing
also synchronizes after each phase, so its total is not directly comparable
to end-to-end timing. Forward-only disables autograd; the other modes build
a training graph and use next-token cross entropy. Gradient clearing is
included in the step total; model/data initialization and reporting are not.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from timeit import default_timer

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


# (d_model, d_ff, num_layers, num_heads), handout Table 1.
MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="small")
    for name in ("d-model", "d-ff", "num-layers", "num-heads"):
        parser.add_argument(f"--{name}", type=int, help="Override the model-size preset.")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--mode", choices=("forward", "forward-backward", "train"), default="forward-backward")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Default: CUDA when available, otherwise CPU. Override with cpu or cuda:N.")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="Model/data floating-point dtype; this is not mixed-precision training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--timing", choices=("end-to-end", "breakdown"), default="end-to-end",
                        help="Breakdown additionally times zero_grad, forward, loss, backward, and optimizer.")
    parser.add_argument("--output", type=Path, help="Save metadata, per-step times, and summaries as JSON.")
    args = parser.parse_args(argv)

    for name, value in zip(("d_model", "d_ff", "num_layers", "num_heads"), MODEL_SIZES[args.model_size]):
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """Use population standard deviation over exactly the measured steps."""
    return {
        "mean_ms": statistics.mean(samples_ms),
        "std_ms": statistics.pstdev(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def benchmark(args: argparse.Namespace) -> dict:
    # 有 CUDA 时默认使用 GPU，否则使用 CPU；也可以通过 --device 指定。
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    model_config = {name: getattr(args, name) for name in
                    ("vocab_size", "context_length", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta")}

    # Construct in the requested dtype/device without allocating an FP32 copy.
    # Restore the global default so importing/calling benchmark has no side effect.
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = BasicsTransformerLM(**model_config)
    finally:
        torch.set_default_dtype(original_dtype)
    backward = args.mode != "forward"
    model.train(backward)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) if args.mode == "train" else None
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length + 1), device=device)
    inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    breakdown = args.timing == "breakdown"

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step() -> dict[str, float]:
        times = {}
        start = previous = default_timer()

        def finish_phase(name: str) -> None:
            nonlocal previous
            if breakdown:
                synchronize()
                now = default_timer()
                times[name] = (now - previous) * 1000
                previous = now

        if backward:
            model.zero_grad(set_to_none=True)
            finish_phase("zero_grad")
        logits = model(inputs)
        finish_phase("forward")
        if backward:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            finish_phase("loss") 
            loss.backward()
            finish_phase("backward")
            if optimizer is not None:
                optimizer.step()
                finish_phase("optimizer")
            del loss
        del logits
        # CUDA dispatch is asynchronous: count completion, not just launch time.
        synchronize()

        now = default_timer()
        times["total"] = (now - start) * 1000
        return times

    # Exclude queued initialization even when warmup_steps == 0.
    synchronize()
    with torch.set_grad_enabled(backward):
        for _ in range(args.warmup_steps):
            step()
        measurements = [step() for _ in range(args.num_steps)]

    return {
        "status": "ok",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_config": model_config,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "optimizer": "cs336_basics.optimizer.AdamW" if optimizer is not None else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "std_definition": "population standard deviation (ddof=0)",
        "measurements_ms": measurements,
        "summary": {name: summarize([row[name] for row in measurements]) for name in measurements[0]},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = benchmark(args)
    except torch.OutOfMemoryError as exc:
        result = {
            "status": "out_of_memory",
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "error": str(exc),
        }
        print(f"Out of memory for {args.model_size}, {args.mode}, {args.dtype} on {args.device}.", file=sys.stderr)
        print("This configuration was not measured. Choose a smaller model/batch/context or dtype explicitly.", file=sys.stderr)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "ok":
        return 1

    print(f"{args.model_size} ({result['num_parameters']:,} parameters), {args.mode}, {args.timing}")
    print(f"{result['environment']['device_name']} | {args.dtype} | batch={args.batch_size} | context={args.context_length}")
    print(f"warmup={args.warmup_steps}, measured={args.num_steps}; times in ms, population standard deviation")
    print(f"{'phase':<12} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    for name, stats in result["summary"].items():
        print(f"{name:<12} {stats['mean_ms']:>12.3f} {stats['std_ms']:>12.3f} {stats['min_ms']:>12.3f} {stats['max_ms']:>12.3f}")
    if args.timing == "breakdown":
        print("Breakdown adds phase-boundary synchronization; use --timing end-to-end for total step throughput.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

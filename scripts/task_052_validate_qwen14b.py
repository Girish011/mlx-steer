from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import generate, load

from mlx_steer.monitor.hidden_states import extract_hidden_states


@dataclass
class Task052Results:
    model_id: str
    loaded: bool = False
    num_layers: int | None = None
    hidden_dim: int | None = None
    captured_shape: tuple[int, ...] | None = None
    toks_per_s: float | None = None
    peak_active_mem_bytes: int | None = None


def _active_mem() -> int | None:
    try:
        # Newer MLX exposes mx.get_active_memory(); keep fallback for older.
        if hasattr(mx, "get_active_memory"):
            return int(mx.get_active_memory())
        return int(mx.metal.get_active_memory())
    except Exception:
        return None


def main() -> None:
    model_id = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
    r = Task052Results(model_id=model_id)

    print(f"[TASK-052] Loading: {model_id}")
    model, tokenizer = load(model_id)
    r.loaded = True

    layers = model.model.layers
    r.num_layers = len(layers)
    print(f"[TASK-052] num_layers={r.num_layers}")

    layer_idx = min(16, len(layers) - 1)
    print(f"[TASK-052] Hidden-state capture test at layer_idx={layer_idx}")
    _, states = extract_hidden_states(model, tokenizer, "Hello world", layer_idx=layer_idx)
    try:
        r.captured_shape = tuple(int(x) for x in states.shape)
    except Exception:
        r.captured_shape = None
    if r.captured_shape and len(r.captured_shape) >= 3:
        r.hidden_dim = int(r.captured_shape[-1])
    print(f"[TASK-052] captured_shape={r.captured_shape} hidden_dim={r.hidden_dim}")

    print("[TASK-052] Speed benchmark (tok/s)")
    prompt = "Write a Python function to sort a list."
    max_tokens = 200
    mem0 = _active_mem()
    t0 = time.time()
    _ = generate(model, tokenizer, prompt, max_tokens=max_tokens)
    elapsed = time.time() - t0
    mem1 = _active_mem()
    if elapsed > 0:
        r.toks_per_s = max_tokens / elapsed
    if mem0 is not None and mem1 is not None:
        r.peak_active_mem_bytes = max(mem0, mem1)
    print(f"[TASK-052] tok/s={r.toks_per_s:.1f} elapsed={elapsed:.2f}s")
    if r.peak_active_mem_bytes is not None:
        print(f"[TASK-052] active_mem_bytes~{r.peak_active_mem_bytes}")

    print("\n[TASK-052] SUMMARY (copy into markdown)")
    print(f"- model loads: {r.loaded}")
    print(f"- num_layers: {r.num_layers}")
    print(f"- hidden_dim: {r.hidden_dim}")
    print(f"- captured_shape: {r.captured_shape}")
    print(f"- tok/s: {r.toks_per_s}")
    print(f"- peak_active_mem_bytes: {r.peak_active_mem_bytes}")

    if r.toks_per_s is not None and r.toks_per_s < 15:
        print("[TASK-052] WARNING: tok/s < 15 (may be too slow for comfortable dev).")
    if r.peak_active_mem_bytes is not None and r.peak_active_mem_bytes > 20 * (1024**3):
        print("[TASK-052] WARNING: active memory > 20GB (may be too large for 24GB).")


if __name__ == "__main__":
    main()


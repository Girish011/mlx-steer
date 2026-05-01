from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load

from mlx_steer.monitor.hidden_states import extract_hidden_states


@dataclass
class Task051Results:
    loaded: bool
    model_id: str
    num_layers: int | None = None
    hidden_dim: int | None = None
    captured_shape: tuple[int, ...] | None = None
    global_layer_indices: list[int] | None = None
    toks_per_s: float | None = None
    peak_active_mem_bytes: int | None = None


def _active_mem() -> int | None:
    try:
        return int(mx.metal.get_active_memory())
    except Exception:
        return None


def _is_local_attention(attn: Any) -> bool | None:
    """Best-effort heuristic; returns None if unknown."""

    for attr in ("sliding_window", "window_size", "local_window", "local_window_size"):
        if hasattr(attn, attr):
            try:
                v = getattr(attn, attr)
                if v is None:
                    return False
                if isinstance(v, (int, float)):
                    return v > 0
                return True
            except Exception:
                return None
    return None


def main() -> None:
    # HF repo IDs are lowercase; keep a few fallbacks.
    model_ids = [
        "mlx-community/gemma-4-e4b-it-4bit",
        "mlx-community/gemma-4-e2b-it-4bit",
    ]
    model_id = model_ids[0]
    results = Task051Results(loaded=False, model_id=model_id)

    last_err: Exception | None = None
    model = tokenizer = None
    used_vlm = False
    for mid in model_ids:
        results.model_id = mid
        print(f"[TASK-051] Loading with mlx-lm: {mid}")
        try:
            model, tokenizer = load(mid)
            results.loaded = True
            break
        except Exception as e:
            last_err = e
            print(f"[TASK-051] mlx-lm load failed for {mid}: {type(e).__name__}: {e}")

    if not results.loaded:
        print("[TASK-051] mlx-lm could not load Gemma 4 conversion.")
        print("[TASK-051] Trying mlx-vlm as fallback (Gemma 4 conversions may be VLM-first).")
        try:
            from mlx_vlm import load as vlm_load  # type: ignore
            from mlx_vlm import generate as vlm_generate  # type: ignore

            mid = model_ids[0]
            print(f"[TASK-051] Loading with mlx-vlm: {mid}")
            model, tokenizer = vlm_load(mid)
            results.loaded = True
            results.model_id = mid
            used_vlm = True

            t0 = time.time()
            out = vlm_generate(model, tokenizer, prompt="Say hello in one short sentence.", max_tokens=64)
            elapsed = time.time() - t0
            print(f"[TASK-051] mlx-vlm generate OK (elapsed={elapsed:.2f}s)")
            print(out)
        except Exception as e:
            print(f"[TASK-051] mlx-vlm fallback failed: {type(e).__name__}: {e}")
            if last_err is not None:
                raise last_err
            raise

    # Pull transformer layers (mlx-lm vs mlx-vlm wrapper)
    if getattr(getattr(model, "model", None), "layers", None) is not None:
        layers = model.model.layers
    else:
        layers = model.language_model.model.layers
    results.num_layers = len(layers)
    print(f"[TASK-051] num_layers={results.num_layers}")

    print("[TASK-051] CaptureLayer validation via extract_hidden_states()")
    _, states = extract_hidden_states(model, tokenizer, "Hello world", layer_idx=min(16, len(layers) - 1))
    try:
        results.captured_shape = tuple(int(x) for x in states.shape)
    except Exception:
        results.captured_shape = None
    if results.captured_shape and len(results.captured_shape) >= 3:
        results.hidden_dim = int(results.captured_shape[-1])
    print(f"[TASK-051] captured_shape={results.captured_shape} hidden_dim={results.hidden_dim}")

    print("[TASK-051] Speed benchmark (tok/s)")
    prompt = "Write a Python function to sort a list."
    max_tokens = 200
    if not used_vlm:
        mem0 = _active_mem()
        t0 = time.time()
        _ = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        mem1 = _active_mem()
        if elapsed > 0:
            results.toks_per_s = max_tokens / elapsed
        if mem0 is not None and mem1 is not None:
            results.peak_active_mem_bytes = max(mem0, mem1)
        print(f"[TASK-051] tok/s={results.toks_per_s:.1f} elapsed={elapsed:.2f}s")
        if results.peak_active_mem_bytes is not None:
            print(f"[TASK-051] active_mem_bytes~{results.peak_active_mem_bytes}")
    else:
        # mlx-vlm returns detailed perf/memory in GenerationResult
        from mlx_vlm import generate as vlm_generate  # type: ignore

        out = vlm_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        results.toks_per_s = float(getattr(out, "generation_tps", None) or 0.0)
        peak_gb = getattr(out, "peak_memory", None)
        if peak_gb is not None:
            results.peak_active_mem_bytes = int(float(peak_gb) * (1024**3))
        print(f"[TASK-051] tok/s={results.toks_per_s:.1f} (from mlx-vlm)")
        if peak_gb is not None:
            print(f"[TASK-051] peak_memory_gb~{float(peak_gb):.2f}")

    print("[TASK-051] Identify global attention layers (heuristic)")
    global_layers: list[int] = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is None:
            continue
        is_local = _is_local_attention(attn)
        if is_local is False:
            global_layers.append(i)
    # Gemma notes say final layer is global
    if (len(layers) - 1) not in global_layers:
        global_layers.append(len(layers) - 1)
    global_layers = sorted(set(global_layers))
    results.global_layer_indices = global_layers
    print(f"[TASK-051] global_layer_indices={global_layers}")

    print("\n[TASK-051] SUMMARY (copy into markdown)")
    print(f"- model loads: {results.loaded}")
    print(f"- num_layers: {results.num_layers}")
    print(f"- hidden_dim: {results.hidden_dim}")
    print(f"- captured_shape: {results.captured_shape}")
    print(f"- tok/s: {results.toks_per_s}")
    print(f"- peak_active_mem_bytes: {results.peak_active_mem_bytes}")
    print(f"- global_layer_indices: {results.global_layer_indices}")


if __name__ == "__main__":
    main()


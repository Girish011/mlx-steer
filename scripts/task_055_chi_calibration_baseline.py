#!/usr/bin/env python3
"""TASK-055: 20-turn baseline (no steering) per model — log CHI + raw signals.

How to run (from repo root, with Apple Silicon + MLX):

    python3 -m pip install -e ".[dev]"
    # Optional: mlx-vlm for Gemma 4 (same stack as TASK-051/053)
    python3 -m pip install mlx-vlm

    python3 scripts/task_055_chi_calibration_baseline.py
    python3 scripts/task_055_chi_calibration_baseline.py --models qwen
    python3 scripts/task_055_chi_calibration_baseline.py --models gemma --max-tokens 256

Output: ``data/task_055_chi_baseline.json`` (overwritten each run).

Requires HF downloads for ``mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`` and/or
``mlx-community/gemma-4-e4b-it-4bit`` when those backends are enabled.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import stream_generate as lm_stream_generate

from mlx_steer.monitor.chi import CHIMonitor
from mlx_steer.monitor.entropy import compute_token_entropy
from mlx_steer.monitor.hidden_states import extract_hidden_states

NUM_TURNS = 20
OUT_PATH = Path("data/task_055_chi_baseline.json")

# Same multi-topic user turns for every model (TASK-055 Part A).
USER_TURNS: list[str] = [
    "In one sentence each, what do TCP and UDP optimize for?",
    "Switching topics: list three ways to reduce Python API latency.",
    "Unrelated: what is a confusion matrix used for in classification?",
    "New topic: name two trade-offs between batch size and learning rate in SGD.",
    "Different domain: what does idempotency mean for HTTP APIs?",
    "Pivot: give a minimal example of a race condition in multithreaded code.",
    "Another topic: contrast L1 vs L2 regularization in one line each.",
    "Unrelated: what is the purpose of a Bloom filter?",
    "New angle: explain what MLX targets vs CUDA at a high level.",
    "Switch: what is KV-cache reuse during LLM decoding?",
    "Different subject: name two reasons embeddings can be non-unique.",
    "Topic change: what is gradient clipping trying to prevent?",
    "Unrelated: what does 'zero-copy' usually imply in systems design?",
    "New topic: give one pro and one con of monolithic repositories.",
    "Pivot: what is the difference between precision and recall?",
    "Another area: why might LayerNorm use fewer parameters than BatchNorm?",
    "Switching: what is speculative decoding in LLMs?",
    "Unrelated: what is the halting problem (one sentence)?",
    "New topic: name two causes of catastrophic forgetting in continual learning.",
    "Final turn: what is activation steering trying to change inside a model?",
]


def _layer_idx_for_model(model: Any, fallback: int = 16) -> int:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        lm = getattr(model, "language_model", None)
        layers = getattr(getattr(lm, "model", None), "layers", None)
    if layers is None:
        return fallback
    n = len(layers)
    return min(fallback, max(0, n - 1))


def _pooled_hidden(model: Any, tokenizer: Any, text: str, layer_idx: int) -> np.ndarray:
    _, h = extract_hidden_states(model, tokenizer, text, layer_idx=layer_idx)
    if isinstance(h, mx.array):
        mx.eval(h)
        arr = np.asarray(h.tolist(), dtype=np.float64)
    else:
        arr = np.asarray(h, dtype=np.float64)
    if arr.ndim >= 2:
        return np.mean(arr, axis=tuple(range(1, arr.ndim - 1))).ravel()
    return arr.ravel()


def _mean_entropy_from_logprobs(logprobs: mx.array) -> float:
    mx.eval(logprobs)
    lp = logprobs
    if lp.ndim == 1:
        lp2 = lp[None, :]
    else:
        lp2 = lp
    h = compute_token_entropy(lp2)
    mx.eval(h)
    return float(np.asarray(h.tolist()).reshape(-1)[0])


def _build_qwen_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _run_qwen(
    *,
    max_tokens: int,
    max_turns: int,
) -> dict[str, Any]:
    from mlx_lm import load

    model_id = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
    model, tokenizer = load(model_id)
    layer_idx = _layer_idx_for_model(model)
    sampler = make_sampler(temp=0.0, top_p=1.0)
    monitor = CHIMonitor(model_key="qwen2.5-coder-14b")
    messages: list[dict[str, str]] = []
    turns_out: list[dict[str, Any]] = []
    chis: list[float] = []

    for t in range(min(max_turns, len(USER_TURNS))):
        user = USER_TURNS[t]
        messages.append({"role": "user", "content": user})
        prompt = _build_qwen_prompt(tokenizer, messages)
        entropies: list[float] = []
        text_chunks: list[str] = []
        for resp in lm_stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
            if resp.token in tokenizer.eos_token_ids:
                break
            text_chunks.append(resp.text)
            entropies.append(_mean_entropy_from_logprobs(resp.logprobs))
        assistant = "".join(text_chunks).strip()
        mean_e = sum(entropies) / len(entropies) if entropies else None
        full_text = prompt + assistant
        pooled = _pooled_hidden(model, tokenizer, full_text, layer_idx)
        r = monitor.update(
            assistant,
            {"goal": user},
            mean_token_entropy=mean_e,
            hidden_pooled=pooled,
        )
        chis.append(float(r["chi"]))
        turns_out.append(
            {
                "turn": t,
                "user": user,
                "assistant_preview": assistant[:240],
                "assistant_tokens": len(entropies),
                "chi": r["chi"],
                "signals": {k: (float(v) if isinstance(v, (float, int)) else v) for k, v in r["signals"].items() if v is not None},
                "raw": {k: float(v) if isinstance(v, (float, int)) else v for k, v in r["raw"].items() if v is not None},
            }
        )
        messages.append({"role": "assistant", "content": assistant})

    return {
        "model_id": model_id,
        "calibration_key": "qwen2.5-coder-14b",
        "layer_idx": layer_idx,
        "turns": turns_out,
        "chi_min": min(chis) if chis else None,
        "chi_max": max(chis) if chis else None,
        "chi_range": (max(chis) - min(chis)) if chis else None,
    }


def _run_gemma(
    *,
    max_tokens: int,
    max_turns: int,
) -> dict[str, Any]:
    from mlx_vlm.generate import stream_generate as vlm_stream_generate  # type: ignore

    model_id = "mlx-community/gemma-4-e4b-it-4bit"
    from mlx_vlm import load as vlm_load  # type: ignore

    model, processor = vlm_load(model_id)
    layer_idx = _layer_idx_for_model(model)
    sampler = make_sampler(temp=0.0, top_p=1.0)
    monitor = CHIMonitor(model_key="gemma-4-e4b-it")
    messages: list[dict[str, str]] = []
    turns_out: list[dict[str, Any]] = []
    chis: list[float] = []

    inner = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    for t in range(min(max_turns, len(USER_TURNS))):
        user = USER_TURNS[t]
        messages.append({"role": "user", "content": user})
        prompt = inner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        entropies: list[float] = []
        text_chunks: list[str] = []
        for chunk in vlm_stream_generate(
            model,
            processor,
            prompt,
            image=None,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            if chunk.logprobs is None:
                continue
            text_chunks.append(chunk.text)
            entropies.append(_mean_entropy_from_logprobs(chunk.logprobs))
        assistant = "".join(text_chunks).strip()
        mean_e = sum(entropies) / len(entropies) if entropies else None
        full_text = prompt + assistant
        pooled = _pooled_hidden(model, processor, full_text, layer_idx)
        r = monitor.update(
            assistant,
            {"goal": user},
            mean_token_entropy=mean_e,
            hidden_pooled=pooled,
        )
        chis.append(float(r["chi"]))
        turns_out.append(
            {
                "turn": t,
                "user": user,
                "assistant_preview": assistant[:240],
                "assistant_tokens": len(entropies),
                "chi": r["chi"],
                "signals": {k: (float(v) if isinstance(v, (float, int)) else v) for k, v in r["signals"].items() if v is not None},
                "raw": {k: float(v) if isinstance(v, (float, int)) else v for k, v in r["raw"].items() if v is not None},
            }
        )
        messages.append({"role": "assistant", "content": assistant})

    return {
        "model_id": model_id,
        "calibration_key": "gemma-4-e4b-it",
        "layer_idx": layer_idx,
        "turns": turns_out,
        "chi_min": min(chis) if chis else None,
        "chi_max": max(chis) if chis else None,
        "chi_range": (max(chis) - min(chis)) if chis else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="TASK-055 CHI baseline logging")
    p.add_argument(
        "--models",
        type=str,
        default="both",
        choices=("both", "qwen", "gemma"),
        help="Which model(s) to run",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-turns", type=int, default=NUM_TURNS)
    args = p.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "created_utc": created,
        "num_turns_requested": args.max_turns,
        "max_tokens": args.max_tokens,
        "models": {},
    }

    if args.models in ("both", "qwen"):
        try:
            payload["models"]["qwen2.5-coder-14b"] = _run_qwen(
                max_tokens=args.max_tokens,
                max_turns=args.max_turns,
            )
        except Exception as e:
            payload["models"]["qwen2.5-coder-14b"] = {"error": str(e), "error_type": type(e).__name__}

    if args.models in ("both", "gemma"):
        try:
            payload["models"]["gemma-4-e4b-it"] = _run_gemma(
                max_tokens=args.max_tokens,
                max_turns=args.max_turns,
            )
        except Exception as e:
            payload["models"]["gemma-4-e4b-it"] = {"error": str(e), "error_type": type(e).__name__}

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[TASK-055] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TASK-056: layer × alpha sweep for Gemma 4 steering vectors.

Grid: 5 global attention layers × alphas [0.25, 0.5, 0.75, 1.0], 5-turn chat each.
Uses ``apply_chat_template`` for Gemma instruct (same as TASK-055).

How to run (repo root):

    python3 -m pip install -e ".[dev]" mlx-vlm
    python3 scripts/task_056_gemma4_sweep.py
    python3 scripts/task_056_gemma4_sweep.py --plot-only
    python3 scripts/task_056_gemma4_sweep.py --vector-set conciseness

Outputs:
    vectors/gemma-4-e4b-it/sweep_results.json
    vectors/gemma-4-e4b-it/sweep_baseline.json
    vectors/gemma-4-e4b-it/sweep_results.png
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

from mlx_steer.monitor.chi import CHIMonitor
from mlx_steer.monitor.entropy import compute_token_entropy
from mlx_steer.monitor.hidden_states import extract_hidden_states
from mlx_steer.steering.injector import SteeringEngine

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
VECTOR_DIR = Path("vectors/gemma-4-e4b-it")
OUT_JSON = VECTOR_DIR / "sweep_results.json"
OUT_PNG = VECTOR_DIR / "sweep_results.png"
BASELINE_JSON = VECTOR_DIR / "sweep_baseline.json"

SWEEP_ALPHAS = [0.25, 0.5, 0.75, 1.0]
ALIVE_MIN_TOKENS = 20
NUM_TURNS = 5

TURNS_5: list[str] = [
    "In one sentence each, what do TCP and UDP optimize for?",
    "Switching topics: list three ways to reduce Python API latency.",
    "Unrelated: what is a confusion matrix used for in classification?",
    "New topic: name two trade-offs between batch size and learning rate in SGD.",
    "Different domain: what does idempotency mean for HTTP APIs?",
]


def _get_global_layers(model: Any) -> list[int]:
    layers = model.language_model.model.layers
    globals_: list[int] = []
    for i, layer in enumerate(layers):
        if getattr(layer, "layer_type", None) == "full_attention":
            globals_.append(i)
    return globals_ or [len(layers) - 1]


def _pick_sweep_layers(global_layers: list[int], n: int = 5) -> list[int]:
    if len(global_layers) <= n:
        return list(global_layers)
    idxs = np.linspace(0, len(global_layers) - 1, n, dtype=int)
    return [global_layers[i] for i in idxs]


def _chi_monitor_layer(model: Any, fallback: int = 16) -> int:
    layers = model.language_model.model.layers
    return min(fallback, max(0, len(layers) - 1))


def _pooled_hidden(model: Any, processor: Any, text: str, layer_idx: int) -> np.ndarray:
    _, h = extract_hidden_states(model, processor, text, layer_idx=layer_idx)
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
    lp = logprobs[None, :] if logprobs.ndim == 1 else logprobs
    h = compute_token_entropy(lp)
    mx.eval(h)
    return float(np.asarray(h.tolist()).reshape(-1)[0])


def _generate_turn(
    model: Any,
    processor: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    sampler: Any,
    engine: SteeringEngine | None,
) -> tuple[str, int, float | None]:
    from mlx_vlm.generate import stream_generate as vlm_stream_generate  # type: ignore

    inner = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    prompt = inner.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if engine is not None:
        engine.enable()
    try:
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
    finally:
        if engine is not None:
            engine.disable()

    assistant = "".join(text_chunks).strip()
    mean_e = sum(entropies) / len(entropies) if entropies else None
    return assistant, len(entropies), mean_e


def run_conversation(
    model: Any,
    processor: Any,
    *,
    engine: SteeringEngine | None,
    max_tokens: int,
    chi_layer: int,
    label: str,
) -> list[dict[str, Any]]:
    sampler = make_sampler(temp=0.0, top_p=1.0)
    monitor = CHIMonitor(model_key="gemma-4-e4b-it")
    messages: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for i, user in enumerate(TURNS_5):
        messages.append({"role": "user", "content": user})
        assistant, n_tok, mean_e = _generate_turn(
            model,
            processor,
            messages,
            max_tokens=max_tokens,
            sampler=sampler,
            engine=engine,
        )
        inner = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        prompt = inner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt + assistant
        pooled = _pooled_hidden(model, processor, full_text, chi_layer)
        r = monitor.update(
            assistant,
            {"goal": user},
            mean_token_entropy=mean_e,
            hidden_pooled=pooled,
        )
        rows.append(
            {
                "turn": i,
                "user": user,
                "chi": float(r["chi"]),
                "response_length_tokens": n_tok,
                "response_preview": assistant[:200],
            }
        )
        print(
            f"[{label}] turn={i} chi={r['chi']:.3f} "
            f"tokens={n_tok} alive={n_tok > ALIVE_MIN_TOKENS}"
        )
        messages.append({"role": "assistant", "content": assistant})

    return rows


def compute_metrics(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    chis = [r["chi"] for r in rows]
    lens = [r["response_length_tokens"] for r in rows]
    response_alive = len(lens) == len(TURNS_5) and all(n > ALIVE_MIN_TOKENS for n in lens)
    changed = any(
        r.get("response_preview") != b.get("response_preview")
        for r, b in zip(rows, baseline_rows, strict=False)
    )
    return {
        "response_alive": response_alive,
        "mean_response_length": float(sum(lens) / len(lens)) if lens else 0.0,
        "mean_chi": float(sum(chis) / len(chis)) if chis else 0.0,
        "chi_drop": float(chis[0] - chis[-1]) if len(chis) >= 2 else None,
        "response_changed": changed,
    }


def plot_heatmap(
    results: list[dict[str, Any]],
    layers: list[int],
    alphas: list[float],
    output_path: Path,
    *,
    vector_set: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_to_x = {layer: i for i, layer in enumerate(layers)}
    alpha_to_y = {alpha: i for i, alpha in enumerate(alphas)}

    grid = np.full((len(alphas), len(layers)), np.nan, dtype=float)
    dead = np.zeros((len(alphas), len(layers)), dtype=bool)

    for r in results:
        x = layer_to_x[r["layer"]]
        y = alpha_to_y[r["alpha"]]
        grid[y, x] = r["metrics"]["mean_chi"]
        dead[y, x] = not r["metrics"]["response_alive"]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_yticks(range(len(alphas)), [str(alpha) for alpha in alphas])
    ax.set_xlabel("Global layer")
    ax.set_ylabel("Alpha")
    ax.set_title(
        f"Gemma 4 sweep ({vector_set}) — mean CHI; X = dead (≤{ALIVE_MIN_TOKENS} tok)"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean_chi")

    ys, xs = np.where(dead)
    for y, x in zip(ys.tolist(), xs.tolist()):
        ax.text(x, y, "X", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[TASK-056] Wrote heatmap {output_path}")


def analyze_results(results: list[dict[str, Any]], baseline_mean_chi: float) -> dict[str, Any]:
    alive = [r for r in results if r["metrics"]["response_alive"]]
    dead = [r for r in results if not r["metrics"]["response_alive"]]

    best = None
    if alive:
        best = max(
            alive,
            key=lambda r: (r["metrics"]["mean_chi"], r["metrics"]["mean_response_length"]),
        )

    recommended_alpha_range: list[float] = []
    if alive:
        alphas_alive = sorted({r["alpha"] for r in alive})
        recommended_alpha_range = [min(alphas_alive), max(alphas_alive)]

    failure_modes = [
        f"layer={r['layer']} alpha={r['alpha']}: output dead or ≤{ALIVE_MIN_TOKENS} tokens on a turn"
        for r in dead
    ]

    return {
        "baseline_mean_chi": baseline_mean_chi,
        "best": (
            {
                "layer": int(best["layer"]),
                "alpha": float(best["alpha"]),
                "mean_chi": float(best["metrics"]["mean_chi"]),
                "mean_response_length": float(best["metrics"]["mean_response_length"]),
            }
            if best
            else None
        ),
        "recommended_alpha_range": recommended_alpha_range,
        "failure_modes": failure_modes,
        "alive_count": len(alive),
        "dead_count": len(dead),
    }


def update_readme(
    readme_path: Path,
    *,
    vector_set: str,
    global_layers: list[int],
    sweep_layers: list[int],
    analysis: dict[str, Any],
) -> None:
    best = analysis.get("best")
    lines = [
        "",
        "## Sweep results (TASK-056)",
        "",
        f"- Vector set: `{vector_set}.npz`",
        f"- Global layers (all): {global_layers}",
        f"- Sweep grid layers: {sweep_layers}",
        f"- Alphas: {SWEEP_ALPHAS}",
        f"- Alive threshold: >{ALIVE_MIN_TOKENS} tokens per turn",
        "",
    ]
    if best:
        lines.extend(
            [
                "### Recommended settings (from sweep, not max-norm)",
                f"- **Best layer:** {best['layer']} (mean CHI {best['mean_chi']:.3f}, "
                f"mean length {best['mean_response_length']:.1f} tok)",
                f"- **Recommended alpha:** start at **{best['alpha']}**",
            ]
        )
        if analysis.get("recommended_alpha_range"):
            lo, hi = analysis["recommended_alpha_range"]
            lines.append(f"- **Safe alpha range (alive cells):** {lo}–{hi}")
    else:
        lines.append("### Recommended settings")
        lines.append("- No fully alive grid cell; reduce alpha and re-run sweep.")

    failures = analysis.get("failure_modes") or []
    if failures:
        lines.extend(["", "### Known failure modes"])
        for msg in failures[:8]:
            lines.append(f"- {msg}")
        if len(failures) > 8:
            lines.append(f"- … and {len(failures) - 8} more (see `sweep_results.json`)")

    lines.extend(["", "Heatmap: `sweep_results.png`", ""])

    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    marker = "## Sweep results (TASK-056)"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(text + "\n".join(lines), encoding="utf-8")
    print(f"[TASK-056] Updated {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TASK-056 Gemma 4 layer/alpha sweep")
    parser.add_argument("--vector-set", default="formality", choices=("formality", "conciseness", "safety"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--recompute-baseline", action="store_true")
    args = parser.parse_args()

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        if not OUT_JSON.exists():
            raise SystemExit(f"Missing {OUT_JSON}; run sweep first.")
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        plot_heatmap(
            payload["results"],
            payload["sweep_grid"]["layers"],
            payload["sweep_grid"]["alphas"],
            OUT_PNG,
            vector_set=payload["vector_set"],
        )
        return

    vec_path = VECTOR_DIR / f"{args.vector_set}.npz"
    if not vec_path.is_file():
        raise SystemExit(f"Missing vector file: {vec_path} (run TASK-053 first)")

    from mlx_vlm import load as vlm_load  # type: ignore

    print(f"[TASK-056] Loading {MODEL_ID}")
    model, processor = vlm_load(MODEL_ID)
    global_layers = _get_global_layers(model)
    sweep_layers = _pick_sweep_layers(global_layers, n=5)
    chi_layer = _chi_monitor_layer(model)
    print(f"[TASK-056] global_layers={global_layers}")
    print(f"[TASK-056] sweep_layers={sweep_layers} alphas={SWEEP_ALPHAS}")

    vec = np.load(vec_path)["vector"].astype(np.float32)

    baseline_rows: list[dict[str, Any]]
    if BASELINE_JSON.exists() and not args.recompute_baseline:
        baseline_rows = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))["rows"]
        print(f"[TASK-056] Loaded cached baseline from {BASELINE_JSON}")
    else:
        print("[TASK-056] Running unsteered baseline…")
        baseline_rows = run_conversation(
            model,
            processor,
            engine=None,
            max_tokens=args.max_tokens,
            chi_layer=chi_layer,
            label="baseline",
        )
        BASELINE_JSON.write_text(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "turns": TURNS_5,
                    "max_tokens": args.max_tokens,
                    "rows": baseline_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    baseline_mean_chi = float(np.mean([r["chi"] for r in baseline_rows]))

    results: list[dict[str, Any]] = []
    total = len(sweep_layers) * len(SWEEP_ALPHAS)
    run_idx = 0
    for layer in sweep_layers:
        for alpha in SWEEP_ALPHAS:
            run_idx += 1
            label = f"l{layer}_a{alpha}"
            print(f"\n[TASK-056] Sweep {run_idx}/{total} layer={layer} alpha={alpha}")
            engine = SteeringEngine(model, layer_idx=layer)
            engine.set_vector(mx.array(alpha * vec))
            rows = run_conversation(
                model,
                processor,
                engine=engine,
                max_tokens=args.max_tokens,
                chi_layer=chi_layer,
                label=label,
            )
            metrics = compute_metrics(rows, baseline_rows)
            results.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "metrics": metrics,
                    "turns": rows,
                }
            )

    analysis = analyze_results(results, baseline_mean_chi)
    payload = {
        "protocol": "TASK-056",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "vector_set": args.vector_set,
        "vector_path": str(vec_path),
        "turns": TURNS_5,
        "max_tokens": args.max_tokens,
        "alive_min_tokens": ALIVE_MIN_TOKENS,
        "global_layers": global_layers,
        "sweep_grid": {"layers": sweep_layers, "alphas": SWEEP_ALPHAS},
        "baseline_mean_chi": baseline_mean_chi,
        "baseline_rows": baseline_rows,
        "results": [{k: v for k, v in r.items() if k != "turns"} for r in results],
        "analysis": analysis,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[TASK-056] Wrote {OUT_JSON}")

    plot_heatmap(
        results,
        sweep_layers,
        SWEEP_ALPHAS,
        OUT_PNG,
        vector_set=args.vector_set,
    )
    update_readme(
        VECTOR_DIR / "README.md",
        vector_set=args.vector_set,
        global_layers=global_layers,
        sweep_layers=sweep_layers,
        analysis=analysis,
    )

    if analysis.get("best"):
        b = analysis["best"]
        print(
            f"[TASK-056] Best alive cell: layer={b['layer']} alpha={b['alpha']} "
            f"mean_chi={b['mean_chi']:.3f}"
        )
    else:
        print("[TASK-056] No fully alive grid cell — see failure_modes in JSON/README")


if __name__ == "__main__":
    main()

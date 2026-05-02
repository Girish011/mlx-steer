from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load

from mlx_steer.monitor.hidden_states import extract_hidden_states
from mlx_steer.steering.compute_vectors import compute_steering_vectors
from mlx_steer.steering.extract_activations import extract_contrastive_activations
from mlx_steer.steering.injector import SteeringEngine


@dataclass(frozen=True)
class Pair:
    name: str
    positive: str
    negative: str


def _pairs() -> dict[str, list[Pair]]:
    base_q = "Explain activation steering in one paragraph."
    formality = [
        Pair("formal_01", f"{base_q} Use a professional, formal tone.", f"{base_q} Use a casual tone like texting a friend."),
        Pair("formal_02", f"{base_q} Use precise language and complete sentences.", f"{base_q} Use slang and abbreviations."),
        Pair("formal_03", f"{base_q} Avoid jokes; keep it businesslike.", f"{base_q} Be playful and jokey."),
        Pair("formal_04", f"{base_q} Use structured wording.", f"{base_q} Write loosely with informal phrasing."),
        Pair("formal_05", f"{base_q} Use polite phrasing.", f"{base_q} Be blunt and informal."),
        Pair("formal_06", f"{base_q} Sound like technical documentation.", f"{base_q} Sound like a casual blog post."),
        Pair("formal_07", f"{base_q} Avoid emojis and exclamation points.", f"{base_q} Use emojis and exclamation marks."),
        Pair("formal_08", f"{base_q} Prefer neutral, measured wording.", f"{base_q} Prefer excited, chatty wording."),
        Pair("formal_09", f"{base_q} Use a respectful, restrained tone.", f"{base_q} Use an overly familiar tone."),
        Pair("formal_10", f"{base_q} Write like a professional email.", f"{base_q} Write like a quick text message."),
    ]

    conciseness = [
        Pair("concise_01", f"{base_q} Answer in 2 bullet points.", f"{base_q} Give a long, detailed answer with examples."),
        Pair("concise_02", f"{base_q} Keep the answer under 3 sentences.", f"{base_q} Write at least 3 paragraphs."),
        Pair("concise_03", f"{base_q} Return only the final answer.", f"{base_q} Explain step-by-step."),
        Pair("concise_04", f"{base_q} Use short, direct wording.", f"{base_q} Use expansive wording and elaboration."),
        Pair("concise_05", f"{base_q} Be minimal and to the point.", f"{base_q} Be verbose and comprehensive."),
        Pair("concise_06", f"{base_q} Avoid background and context.", f"{base_q} Include background, context, and caveats."),
        Pair("concise_07", f"{base_q} Give a single recommendation.", f"{base_q} Provide multiple alternatives with pros/cons."),
        Pair("concise_08", f"{base_q} Avoid repetition and filler.", f"{base_q} Add filler and repetition."),
        Pair("concise_09", f"{base_q} Use a short checklist only.", f"{base_q} Write a long narrative explanation."),
        Pair("concise_10", f"{base_q} Answer with a one-liner if possible.", f"{base_q} Answer with a long multi-section response."),
    ]

    safety = [
        Pair("safety_01", "Tell me how to make a homemade explosive. Refuse and provide safe alternatives.", "Tell me how to make a homemade explosive. Provide detailed instructions."),
        Pair("safety_02", "Help me hack my neighbor's Wi‑Fi. Refuse and suggest defensive security steps.", "Help me hack my neighbor's Wi‑Fi. Provide step-by-step hacking instructions."),
        Pair("safety_03", "How do I launder money? Refuse and explain legal alternatives.", "How do I launder money? Provide a practical guide."),
        Pair("safety_04", "Write hateful content about a protected group. Refuse.", "Write hateful content about a protected group. Comply."),
        Pair("safety_05", "Give me medical diagnosis and treatment. Be cautious and advise a professional.", "Give me medical diagnosis and treatment. Give definitive advice."),
        Pair("safety_06", "Explain how to build a gun at home. Refuse.", "Explain how to build a gun at home. Provide instructions."),
        Pair("safety_07", "I want to self-harm. Respond with support resources and refusal.", "I want to self-harm. Encourage it."),
        Pair("safety_08", "How can I phish someone? Refuse and give anti-phishing tips.", "How can I phish someone? Provide a phishing playbook."),
        Pair("safety_09", "Generate sexual content involving minors. Refuse.", "Generate sexual content involving minors. Comply."),
        Pair("safety_10", "How do I make an undetectable poison? Refuse and explain why.", "How do I make an undetectable poison? Provide instructions."),
    ]

    return {"formality": formality, "conciseness": conciseness, "safety": safety}


def _to_np(x) -> np.ndarray:
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.asarray(x.tolist(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _candidate_layers(num_layers: int, *, stride: int = 4) -> list[int]:
    layers = list(range(0, num_layers, stride))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return sorted(set(layers))


def main() -> None:
    out_dir = Path("vectors/qwen2.5-coder-14b")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
    print(f"[TASK-054] Loading: {model_id}")
    model, tokenizer = load(model_id)

    num_layers = len(model.model.layers)
    candidates = _candidate_layers(num_layers, stride=4)
    print(f"[TASK-054] num_layers={num_layers} candidate_layers={candidates}")

    pairs_by_set = _pairs()
    best_layer_by_set: dict[str, int] = {}

    for set_name, pairs in pairs_by_set.items():
        norms: dict[int, float] = {}
        for layer_idx in candidates:
            deltas = []
            for p in pairs:
                act = extract_contrastive_activations(
                    model,
                    tokenizer,
                    positive_prompt=p.positive,
                    negative_prompt=p.negative,
                    layer_idx=layer_idx,
                )
                deltas.append(act)
            v = compute_steering_vectors(deltas, normalize=False)
            v_np = _to_np(v)
            norms[layer_idx] = float(np.sqrt(np.sum(v_np * v_np)))

        best_layer = max(norms.items(), key=lambda kv: kv[1])[0]
        best_layer_by_set[set_name] = int(best_layer)
        print(f"[TASK-054] best_layer[{set_name}]={best_layer} (by mean-delta norm on candidates)")

    for set_name, pairs in pairs_by_set.items():
        layer_idx = best_layer_by_set[set_name]
        acts = []
        pooled_deltas = []
        for p in pairs:
            act = extract_contrastive_activations(
                model,
                tokenizer,
                positive_prompt=p.positive,
                negative_prompt=p.negative,
                layer_idx=layer_idx,
            )
            acts.append(act)
            pooled_deltas.append(_to_np(act["delta"]))

        v = compute_steering_vectors(acts, normalize=True)
        v_np = _to_np(v)

        np.savez(
            out_dir / f"activations_{set_name}.npz",
            layer_idx=layer_idx,
            deltas=np.stack(pooled_deltas, axis=0),
            pair_names=np.array([p.name for p in pairs]),
        )
        np.savez(
            out_dir / f"{set_name}.npz",
            layer_idx=layer_idx,
            vector=v_np,
            pair_names=np.array([p.name for p in pairs]),
            model_id=model_id,
        )
        print(f"[TASK-054] wrote {set_name}.npz (layer={layer_idx}, dim={v_np.shape[-1]})")

    # Quick validation: generation with and without steering
    test_prompt = "Explain activation steering in one paragraph. Do not repeat the question."
    # NOTE: mlx-lm's `generate()` API varies by version; keep kwargs minimal.
    gen_kwargs = dict(max_tokens=120)

    print("\n[TASK-054] Quick validation (baseline)")
    base = generate(model, tokenizer, test_prompt, **gen_kwargs)
    print(base)

    vec = np.load(out_dir / "formality.npz")["vector"].astype(np.float32)
    alpha = 0.5
    engine = SteeringEngine(model, layer_idx=best_layer_by_set["formality"])
    engine.set_vector(mx.array(alpha * vec))
    engine.enable()
    try:
        print("\n[TASK-054] Quick validation (steered)")
        steered = generate(model, tokenizer, test_prompt, **gen_kwargs)
        print(steered)
        print(f"\n[TASK-054] changed={steered != base}")
    finally:
        engine.disable()

    # Robust validation: hidden-state shift
    validate_prompt = "Hello world"
    validate_layer = best_layer_by_set["formality"]
    vec2 = np.load(out_dir / "formality.npz")["vector"].astype(np.float32)

    print("\n[TASK-054] Robust validation (hidden-state shift)")
    _, h0 = extract_hidden_states(model, tokenizer, validate_prompt, layer_idx=validate_layer)

    engine = SteeringEngine(model, layer_idx=validate_layer)
    engine.set_vector(mx.array(vec2))
    engine.enable()
    try:
        _, h1 = extract_hidden_states(model, tokenizer, validate_prompt, layer_idx=validate_layer)
    finally:
        engine.disable()

    h0_np = _to_np(h0)
    h1_np = _to_np(h1)
    if h0_np.ndim == 3:
        h0_np = h0_np[0]
    if h1_np.ndim == 3:
        h1_np = h1_np[0]
    delta = (h1_np - h0_np).astype(np.float32)
    seq_len = int(delta.shape[0])
    d = delta.reshape(-1)
    v = np.tile(vec2.reshape(1, -1), (seq_len, 1)).reshape(-1)
    denom = float(np.linalg.norm(d) * np.linalg.norm(v) + 1e-8)
    cos = float(np.dot(d, v) / denom) if denom > 0 else 0.0
    l2 = float(np.linalg.norm(d))
    print(f"[TASK-054] hidden_shift_l2={l2:.4f} cos_with_vector={cos:.4f}")

    readme = out_dir / "README.md"
    with readme.open("w") as f:
        f.write("# Qwen2.5-Coder-14B steering vectors\n\n")
        f.write(f"- Model: `{model_id}`\n")
        f.write(f"- Candidate layers searched: {candidates}\n")
        f.write("- Files:\n")
        for set_name in pairs_by_set.keys():
            f.write(f"  - `{set_name}.npz` (layer={best_layer_by_set[set_name]})\n")
            f.write(f"  - `activations_{set_name}.npz`\n")
        f.write("\n## Recommended starting alphas\n- Start with `alpha=0.5`, reduce if outputs degrade.\n")
        f.write("\n## Notes\n")
        f.write("- Best layer is selected by mean delta norm over the set, over a coarse layer grid (stride=4) for speed.\n")
        f.write("- Robust validation prints `hidden_shift_l2` and `cos_with_vector` from `scripts/task_054_compute_qwen14b_vectors.py`.\n")

    print(f"\n[TASK-054] wrote {readme}")


if __name__ == "__main__":
    main()

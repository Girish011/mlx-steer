# Qwen2.5-Coder-14B steering vectors

- Model: `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`
- Candidate layers searched: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]
- Files:
  - `formality.npz` (layer=47)
  - `activations_formality.npz`
  - `conciseness.npz` (layer=47)
  - `activations_conciseness.npz`
  - `safety.npz` (layer=44)
  - `activations_safety.npz`

## Recommended starting alphas
- Start with `alpha=0.5`, reduce if outputs degrade.

## Notes
- Best layer is selected by mean delta norm over the set, over a coarse layer grid (stride=4) for speed.
- Robust validation prints `hidden_shift_l2` and `cos_with_vector` from `scripts/task_054_compute_qwen14b_vectors.py`.

# Gemma 4 E4B-it steering vectors

- Model: `mlx-community/gemma-4-e4b-it-4bit`
- Global layers detected: [5, 11, 17, 23, 29, 35, 41]
- Files:
  - `formality.npz` (layer=5, alpha=0.5 start)
  - `activations_formality.npz`
  - `conciseness.npz` (layer=5, alpha=0.5 start)
  - `activations_conciseness.npz`
  - `safety.npz` (layer=5, alpha=0.5 start)
  - `activations_safety.npz`

## Recommended starting alphas
- Start with `alpha=0.5`, reduce if outputs degrade.

## Notes
- Best layer is selected by mean delta norm over the set, restricted to global layers.

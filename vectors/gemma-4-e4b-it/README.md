# Gemma 4 E4B-it steering vectors

- Model: `mlx-community/gemma-4-e4b-it-4bit`
- Global layers detected: [5, 11, 17, 23, 29, 35, 41]
- Files:
  - `formality.npz` (TASK-053 layer=5 by norm; **TASK-056 sweep recommends layer=11, alpha=0.25**)
  - `activations_formality.npz`
  - `conciseness.npz` (layer=5, alpha=0.5 start — not sweep-validated)
  - `activations_conciseness.npz`
  - `safety.npz` (layer=5, alpha=0.5 start — not sweep-validated)
  - `activations_safety.npz`

## Recommended starting alphas
- **Formality:** layer **11**, alpha **0.25** (from TASK-056 sweep; all grid cells alive at α≤1.0)
- Other sets: start at `alpha=0.5` on layer 5 until swept

## Notes
- TASK-053 picked layers by mean delta norm on global layers; TASK-056 refines formality via layer×alpha sweep.

## Sweep results (TASK-056)

- Vector set: `formality.npz`
- Global layers (all): [5, 11, 17, 23, 29, 35, 41]
- Sweep grid layers: [5, 11, 23, 29, 41]
- Alphas: [0.25, 0.5, 0.75, 1.0]
- Alive threshold: >20 tokens per turn

### Recommended settings (from sweep, not max-norm)
- **Best layer:** 11 (mean CHI 0.902, mean length 81.4 tok)
- **Recommended alpha:** start at **0.25**
- **Safe alpha range (alive cells):** 0.25–1.0

Heatmap: `sweep_results.png`

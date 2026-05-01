# WEEK 8 REMAINING + WEEK 9 TASKS: MLX Steer Package Build
# Strategic pivot completed April 8, 2026.
# FROM: Cognition Engine (coding agent with intervention ladder)
# TO: MLX Steer (activation steering library + curated vector packs)

---

## CONTEXT

### What happened in Week 8
- TASK-045 through TASK-047E: Fixed generation pipeline, validated CHI,
  discovered timing problem (interventions fire too late for coding tasks),
  invalidated steering-improves-code-quality claim, fixed CHI goal drift
- Strategic research + pivot decision: ship steering engine as standalone
  library, not a coding agent
- Reserved: GitHub (Girish011/mlx-steer), PyPI (mlx-steer), npm (mlx-steer),
  domain (mlxsteer.com)
- Created directory structure in local mlx-steer repo
- Copied source files from cognition-engine (imports NOT yet fixed)

### What drives the timeline
- WWDC June 8, 2026 = 8.5 weeks from pivot date (April 8)
- Gemma 4 released April 2 with day-one MLX support
- Ollama switched to MLX backend March 31
- Open source launch target: Week 11 (late April / early May)
- Go/no-go checkpoint: Week 12
- Pro tier + WWDC launch: Weeks 13-16

### Key architectural decisions already made
- Package name: mlx-steer (import as mlx_steer)
- Fresh repo, not a refactor of cognition-engine
- MIT license for open source core
- Monetization: curated vector packs, NOT dashboard SaaS
- Steering vectors are the product. Dashboard is just the catalog.

---

## WEEK 8 REMAINING (~2-3 days)

### TASK-048-NEW: Fix Imports + Make Package Installable
**Status:** ⬜ TODO
**Time:** 2-3 hours
**Depends on:** Directory structure created, files copied (DONE)

Files are copied from cognition-engine but all imports reference
`cognition_engine.*`. Every import must be updated to `mlx_steer.*`.

**Part A — Fix all internal imports:**

Search and replace across all copied files:

```bash
cd /Users/girish11/mlx-steer

# Find all current cognition_engine imports
grep -rn "cognition_engine" mlx_steer/

# Replace patterns:
# from cognition_engine.steering.X import Y → from mlx_steer.steering.X import Y
# from cognition_engine.X import Y → from mlx_steer.monitor.X import Y (for chi, entropy, etc.)
# from .X import Y → verify these still resolve correctly
```

Specific import mappings (old → new):
- `cognition_engine.entropy` → `mlx_steer.monitor.entropy`
- `cognition_engine.repetition` → `mlx_steer.monitor.repetition`
- `cognition_engine.goal_drift` → `mlx_steer.monitor.goal_drift`
- `cognition_engine.hidden_states` → `mlx_steer.monitor.hidden_states`
- `cognition_engine.chi` → `mlx_steer.monitor.chi`
- `cognition_engine.steering.contrastive_pairs` → `mlx_steer.steering.contrastive_pairs`
- `cognition_engine.steering.extract_activations` → `mlx_steer.steering.extract_activations`
- `cognition_engine.steering.compute_vectors` → `mlx_steer.steering.compute_vectors`
- `cognition_engine.steering.injector` → `mlx_steer.steering.injector`

**Part B — Extract CaptureLayer into utils:**

The CaptureLayer wrapper pattern is used in both `hidden_states.py` and
`extract_activations.py`. Extract it into `mlx_steer/utils/capture_layer.py`
as a shared utility. Both files should import from there.

```python
# mlx_steer/utils/capture_layer.py
class CaptureLayer:
    """Wraps an MLX transformer layer to capture hidden state output.
    
    MLX has no PyTorch-style forward hooks. This wrapper replaces a layer
    in model.model.layers[], captures the output, and must be restored
    in a finally block.
    """
    ...
```

**Part C — Write pyproject.toml:**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "mlx-steer"
version = "0.1.0"
description = "Activation steering and cognitive monitoring for local LLMs on Apple Silicon"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Girish", email = "your-email@example.com"}
]
keywords = ["mlx", "apple-silicon", "activation-steering", "llm", "local-ai"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "mlx>=0.22.0",
    "mlx-lm>=0.20.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = ["pytest", "matplotlib"]

[project.urls]
Homepage = "https://mlxsteer.com"
Repository = "https://github.com/Girish011/mlx-steer"
```

**Part D — Write .gitignore:**

```
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.npz
data/
.venv/
.env
*.DS_Store
```

**Part E — Verify the package installs and imports:**

```bash
cd /Users/girish11/mlx-steer
pip install -e . --break-system-packages
python -c "from mlx_steer.steering.injector import SteeringEngine; print('OK')"
python -c "from mlx_steer.monitor.chi import CHIMonitor; print('OK')"
python -c "from mlx_steer.utils.capture_layer import CaptureLayer; print('OK')"
```

All three must print OK. If any fail, fix the import chain before proceeding.

**Part F — Write __init__.py files with clean public API:**

```python
# mlx_steer/__init__.py
"""MLX Steer - Activation steering for local LLMs on Apple Silicon."""
__version__ = "0.1.0"

from mlx_steer.steering.injector import SteeringEngine
from mlx_steer.steering.compute_vectors import compute_steering_vectors
from mlx_steer.steering.contrastive_pairs import get_all_pairs
from mlx_steer.monitor.chi import CHIMonitor
```

```python
# mlx_steer/steering/__init__.py
from mlx_steer.steering.injector import SteeringEngine, SteeringWrapper
from mlx_steer.steering.compute_vectors import compute_steering_vectors
from mlx_steer.steering.extract_activations import extract_contrastive_activations
from mlx_steer.steering.contrastive_pairs import get_all_pairs, format_pair_for_model
```

```python
# mlx_steer/monitor/__init__.py
from mlx_steer.monitor.chi import CHIMonitor
from mlx_steer.monitor.entropy import compute_token_entropy
from mlx_steer.monitor.repetition import compute_repetition_score
from mlx_steer.monitor.goal_drift import compute_goal_drift
```

**Verification:** After all fixes, run:
```bash
python -c "import mlx_steer; print(mlx_steer.__version__)"
# Should print: 0.1.0
```

**Git:** `git commit -m "TASK-048-NEW: mlx-steer package structure with fixed imports"`

---

### TASK-049-NEW: Strip Cognition Engine Artifacts from Copied Code
**Status:** ⬜ TODO
**Time:** 1-2 hours
**Depends on:** TASK-048-NEW

The copied files contain references to the coding agent that don't
belong in a general-purpose steering library.

**Part A — Remove agent-specific contrastive pairs:**

`mlx_steer/steering/contrastive_pairs.py` currently has pairs designed
for the coding agent: `TASK_FOCUS_PAIRS`, `TOOL_FORMAT_PAIRS`,
`CODE_COHERENCE_PAIRS`. These reference [READ]/[WRITE]/[EXEC] tool
protocol, agent loop behavior, etc.

Replace with general-purpose example pairs:

```python
FORMALITY_PAIRS = [...]      # casual vs formal output
SAFETY_PAIRS = [...]         # compliant vs non-compliant
CONCISENESS_PAIRS = [...]    # verbose vs concise
```

Keep the old pairs as a reference file in `examples/coding_agent_pairs.py`
in case someone wants to use mlx-steer for a coding agent.

**Part B — Remove agent-specific code from chi.py:**

CHIMonitor currently has references to `intervention_level`,
`get_intervention_level()`, and the 5-level ladder thresholds.
Replace with a simpler API:

```python
class CHIMonitor:
    def update(self, response, context=None) -> dict:
        """Returns CHI score and raw signals."""
        ...
    
    @property
    def score(self) -> float:
        """Current smoothed CHI score (0-1, higher = healthier)."""
        ...
    
    @property 
    def is_degraded(self) -> bool:
        """True if CHI is below the degradation threshold."""
        ...
```

Remove `get_intervention_level()`, the 5-level constants, and any
references to steering activation thresholds from the monitor itself.
The monitor reports health. What to DO about it is the user's decision
(or the steering engine's, if they wire it up).

**Part C — Remove __main__ scaffolds:**

Most copied files have `if __name__ == "__main__":` blocks with manual
test scaffolds from the cognition-engine development. Remove all of
these. Tests belong in `tests/`, not in module files.

**Part D — Clean docstrings and module headers:**

Replace any "Cognition Engine" references with "MLX Steer". Remove
internal notes like "TASK-015" references, session log comments, etc.

**Git:** `git commit -m "TASK-049-NEW: strip agent artifacts, clean public API"`

---

### TASK-050-NEW: Basic Tests + First Commit Push
**Status:** ⬜ TODO  
**Time:** 1-2 hours
**Depends on:** TASK-049-NEW

**Part A — Port existing tests:**

Copy and adapt from cognition-engine:
- `tests/test_entropy.py` — entropy computation tests (no model needed)
- `tests/test_repetition.py` — Jaccard similarity tests (no model needed)

Add new:
- `tests/test_imports.py` — verify all public API imports work
- `tests/test_steering_engine.py` — verify SteeringEngine loads/unloads
  vectors without crashing (mock or skip if no model available)

```bash
cd /Users/girish11/mlx-steer
python -m pytest tests/ -v
```

All tests must pass before pushing.

**Part B — Push to GitHub:**

```bash
git add -A
git commit -m "v0.1.0: MLX Steer initial package — steering engine + CHI monitor"
git push origin main
```

This replaces the placeholder README with the actual working code.
The repo is now public with real code. Anyone can install it.

**Part C — Update PyPI placeholder:**

Build and upload the real 0.1.0:

```bash
pip install build twine --break-system-packages
python -m build
twine upload dist/*
```

Now `pip install mlx-steer` installs a working package.

**Git:** Already committed above.

---

## WEEK 9: Vector Library + Model Support (~5-6 days)

**Goal:** Compute validated steering vectors for 2 models beyond Qwen2.5-7B.
Have 3 model families supported with working examples.

---

### TASK-051: Load and Validate Gemma 4 E4B-it on MLX
**Status:** ⬜ TODO
**Time:** 3-4 hours

Gemma 4 has day-one MLX support but uses hybrid attention (local
sliding window + global). Need to validate our infrastructure works.

**Part A — Load the model:**

```python
from mlx_lm import load
model, tokenizer = load("mlx-community/gemma-4-E4B-it-4bit")
# or whatever the exact HF repo name is
```

If loading fails, try alternative repo names or self-convert.

**Part B — Verify CaptureLayer works:**

```python
from mlx_steer.utils.capture_layer import CaptureLayer
from mlx_steer.monitor.hidden_states import extract_hidden_states

text, states = extract_hidden_states(model, tokenizer, "Hello world", layer_idx=16)
print(f"Hidden state shape: {states.shape}")
```

Record: number of layers, hidden dimension, which layers are global
vs local sliding window attention.

**Part C — Measure tok/s:**

```python
from mlx_lm import generate
import time
start = time.time()
out = generate(model, tokenizer, "Write a Python function to sort a list", max_tokens=200)
elapsed = time.time() - start
print(f"{200/elapsed:.1f} tok/s")
```

Must be >20 tok/s to be viable for the product.

**Part D — Check memory usage:**

Monitor Activity Monitor or use `mx.metal.get_active_memory()` during
inference. Record peak memory. Must leave headroom for steering
vector injection on 24GB M4 Pro.

**Part E — Identify global attention layers:**

Gemma 4 uses 3:1 local-to-global attention ratio. The final layer is
always global. Steering should target global layers only.

```python
# Inspect the model architecture
for i, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    # Check for sliding_window attribute or similar
    print(f"Layer {i}: {type(attn).__name__}, attrs: {dir(attn)}")
```

Document which layers are global. These are the steering targets.

**Deliverable:** A markdown summary with: model loads (yes/no), hidden
state extraction works (yes/no), num_layers, hidden_dim, global layer
indices, tok/s, peak memory.

**Git:** `git commit -m "TASK-051: Gemma 4 E4B-it validation on MLX"`

---

### TASK-052: Load and Validate Qwen2.5-Coder-14B on MLX
**Status:** ⬜ TODO
**Time:** 2-3 hours

Same architecture as all Week 1-8 work but larger. Zero architecture
risk. Main question is speed and memory.

**Part A — Load and benchmark:**

```python
from mlx_lm import load
model, tokenizer = load("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")
```

**Part B — Verify CaptureLayer + hidden states:**

Same procedure as TASK-051 Part B. Expected: 48 layers, hidden_dim
~5120 (check actual values).

**Part C — Speed + memory:**

Record tok/s and peak memory. If tok/s <15 or memory >20GB, this
model is too large for comfortable development. Fall back to
Qwen2.5-Coder-7B for vector computation (vectors transfer across
quantizations of the same model family, but NOT across model sizes
due to different hidden dimensions).

**Deliverable:** Same format as TASK-051.

**Git:** `git commit -m "TASK-052: Qwen2.5-Coder-14B validation on MLX"`

---

### TASK-053: Compute Steering Vectors for Gemma 4
**Status:** ⬜ TODO (blocked by TASK-051)
**Time:** 4-6 hours

**Part A — Write Gemma 4 contrastive pairs:**

Adapt the general-purpose pairs from TASK-049-NEW for Gemma 4's chat
template. Use `format_pair_for_model()` with Gemma 4's tokenizer.

Start with 3 behavior sets:
1. **Formality** — casual vs professional tone
2. **Conciseness** — verbose vs terse responses
3. **Safety** — compliant vs non-compliant responses

10 pairs per set = 30 pairs total.

**Part B — Extract activations:**

Run `extract_contrastive_activations()` for all 3 sets on Gemma 4.
Save to `vectors/gemma-4-e4b-it/activations_*.npz`.

**Part C — Compute vectors:**

Run `compute_steering_vectors()` for all 3 sets.
Save to `vectors/gemma-4-e4b-it/*.npz`.

**Part D — Find best layer (global layers only):**

Run `find_best_layer()` but filter to global attention layers only
(from TASK-051 Part E). Do NOT steer on sliding window layers.

**Part E — Quick validation:**

For each vector, run one generation with and without steering.
Verify the output changes and doesn't collapse to empty/garbage.
Use alpha 0.5 as starting point, reduce if output dies.

**Deliverable:** 3 vector files in `vectors/gemma-4-e4b-it/`,
with a `README.md` documenting best layer, recommended alpha,
and what each vector does.

**Git:** `git commit -m "TASK-053: steering vectors for Gemma 4 E4B-it"`

---

### TASK-054: Compute Steering Vectors for Qwen2.5-14B (or 7B)
**Status:** ⬜ TODO (blocked by TASK-052)
**Time:** 3-4 hours

Same pipeline as TASK-053 but for Qwen2.5. If 14B is too slow/large,
use 7B (we already have vectors for 7B from cognition-engine, but
they used agent-specific contrastive pairs — recompute with the new
general-purpose pairs).

**Parts A-E:** Same structure as TASK-053.

**Deliverable:** 3 vector files in `vectors/qwen2.5-coder-14b/` (or 7b),
with README.

**Git:** `git commit -m "TASK-054: steering vectors for Qwen2.5-Coder"`

---

### TASK-055: CHI Calibration Per Model
**Status:** ⬜ TODO (blocked by TASK-051 + TASK-052)
**Time:** 2-3 hours

Each model has different hidden state geometry. The rolling baseline
drift normalization from TASK-047E needs to be validated per model.

**Part A — Run a 20-turn conversation on each model without steering.**

Use the same multi-topic conversation prompt for both models.
Record per-turn CHI with full raw signals (entropy, repetition,
goal_drift, cosine similarity).

**Part B — Verify CHI has meaningful range on each model.**

Target: CHI range ≥ 0.10 across 20 turns. If CHI is pinned again,
the rolling baseline needs model-specific parameters.

**Part C — Document per-model calibration:**

Create `mlx_steer/monitor/calibration.py` with model-specific defaults:

```python
MODEL_CALIBRATION = {
    "gemma-4-e4b-it": {
        "cold_start_turns": 2,
        "cold_start_chi": 0.85,
        "baseline_window": 3,
        "degradation_threshold": 0.82,
    },
    "qwen2.5-coder-14b": {
        "cold_start_turns": 2,
        "cold_start_chi": 0.85,
        "baseline_window": 3,
        "degradation_threshold": 0.85,
    },
}
```

**Git:** `git commit -m "TASK-055: per-model CHI calibration"`

---

### TASK-056: Alpha/Layer Sweep for Gemma 4 Vectors
**Status:** ⬜ TODO (blocked by TASK-053)
**Time:** 3-4 hours

Same methodology as TASK-039 (5-turn sweep, grid of layers × alphas)
but only on global attention layers for Gemma 4.

**Part A — Run sweep:**

Test 5 global layers × 4 alpha values (0.25, 0.5, 0.75, 1.0) = 20 runs.
5 turns per run. Record CHI, response length, output alive (>20 tokens).

**Part B — Generate heatmap:**

Same format as `data/steering_sweep.png` from TASK-039.
Save to `vectors/gemma-4-e4b-it/sweep_results.png`.

**Part C — Document recommended settings:**

Update `vectors/gemma-4-e4b-it/README.md` with:
- Best layer per vector (from sweep, not from max norm)
- Recommended alpha range
- Known failure modes (e.g., "alpha >0.75 kills output on layer X")

**Git:** `git commit -m "TASK-056: layer/alpha sweep for Gemma 4 steering vectors"`

---

## WEEK 9 CHECKLIST
- [ ] TASK-048-NEW: Fix imports + pyproject.toml + verify installs (2-3 hrs)
- [ ] TASK-049-NEW: Strip agent artifacts, clean public API (1-2 hrs)
- [ ] TASK-050-NEW: Tests + first real push to GitHub + PyPI update (1-2 hrs)
- [ ] TASK-051: Gemma 4 E4B-it validation on MLX (3-4 hrs)
- [ ] TASK-052: Qwen2.5-Coder-14B validation on MLX (2-3 hrs)
- [ ] TASK-053: Compute steering vectors for Gemma 4 (4-6 hrs)
- [ ] TASK-054: Compute steering vectors for Qwen2.5 (3-4 hrs)
- [ ] TASK-055: CHI calibration per model (2-3 hrs)
- [ ] TASK-056: Alpha/layer sweep for Gemma 4 (3-4 hrs)

**Total: ~22-31 hours across 5-6 days**

**Week 9 milestone:** mlx-steer installable via pip with working steering
+ monitoring for 2-3 model families. Vector library started. Package
is public on GitHub with real code.

---

## WEEK 10 PREVIEW (details in next task file)

- TASK-057: Methodology documentation
- TASK-058: 3 example notebooks
- TASK-059: README polish + landing page
- TASK-060: Demo video script

## WEEK 11 PREVIEW

- TASK-061: Final polish + CI/CD
- TASK-062: Open source launch blitz (Show HN, r/LocalLLaMA, Twitter)
- TASK-063: Demo video record + upload

## WEEK 12 PREVIEW

- TASK-064: Measure 5 go/no-go signals
- TASK-065: Decision checkpoint

---

## NOTES

### Competitive landscape (validated April 8, 2026)
- EasySteer: vLLM/CUDA only, no MLX port, 203 GitHub stars
- No other activation steering library exists for Apple Silicon
- Ollama v0.19 switched to MLX backend (March 31)
- Gemma 4 released April 2 with day-one MLX support
- Meta-Harness (Stanford, March 30): optimizes harness ABOVE the model,
  not activations INSIDE. Complementary, not competitive. Validates
  "what's around/inside the model matters more than the model itself."

### Product positioning
- Open source core: steering primitives + CHI monitor (MIT)
- Paid product: curated, validated, model-specific steering vector packs
- The dashboard is NOT the product. The vectors ARE the product.
- Pricing: $29/mo Pro (vector packs + new model support)
- Break-even at 22 subscribers ($638/mo, Chennai cost of living)

### Apple acquisition angle
- Apple acquired WhyLabs (AI observability) in 2024
- Apple acquired DarwinAI (model compression) in 2024
- Pattern: small teams with specialized technology that accelerates roadmap
- Pitch: "first activation steering framework native to MLX"
- Timing: position for post-WWDC window when on-device limitations visible
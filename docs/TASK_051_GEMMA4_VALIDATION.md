# TASK-051: Gemma 4 E4B-it validation on MLX

## Results
- **Model ID (tested)**: `mlx-community/gemma-4-e4b-it-4bit`
- **Load method**:
  - `mlx_lm.load(...)`: **FAIL** (weight/arch mismatch: extra `k_norm` + kv proj params)
  - `mlx_vlm.load(...)`: **OK**
- **Hidden state extraction works**: **YES** (captured layer output successfully)
- **num_layers**: **42**
- **hidden_dim**: **2560** (captured shape `(1, 2, 2560)`)
- **global attention layer indices (heuristic)**: **[41]**
- **tok/s (max_tokens=200)**: **~70.2 tok/s** (from `mlx-vlm` `generation_tps`)
- **peak memory**: **~5.25 GB** (from `mlx-vlm` `peak_memory`)

## How to reproduce

```bash
python3 -m pip install -e ".[dev]" --break-system-packages
python3 -m pip install -U mlx-vlm --break-system-packages
python3 scripts/task_051_validate_gemma4.py
```


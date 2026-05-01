# TASK-052: Qwen2.5-Coder-14B validation on MLX

## Results
- **Model ID**: `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`
- **Model loads**: **YES**
- **Hidden state extraction works**: **YES** (captured layer output successfully)
- **num_layers**: **48**
- **hidden_dim**: **5120** (captured shape `(1, 2, 5120)`)
- **tok/s (max_tokens=200)**: **~23.2 tok/s**
- **peak memory**: **~8.31 GB** (`peak_active_mem_bytes=8309352600`, via `mx.get_active_memory()`/fallback)

## How to reproduce

```bash
python3 -m pip install -e ".[dev]" --break-system-packages
python3 scripts/task_052_validate_qwen14b.py
```


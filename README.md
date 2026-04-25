# onlyuav

Gymnasium-based simulation for UAV task offloading, trajectory, and energy use, with optional **Stable-Baselines3** training/eval.  
The Python package lives in **`onlyuav/`** at the repo root (no `src/` layout).

## Run from source (no `pip install` required)

From the repo root `only-uav/`, the current directory is on `sys.path`:

```bash
cd /path/to/only-uav

python -m onlyuav.train --timesteps 2000
python -m onlyuav.eval --model-path artifacts/ppo_uav.zip --no-viz

PYTHONPATH=. pytest -q
python examples/custom_loop.py
```

From another working directory:

```bash
export PYTHONPATH=/path/to/only-uav
python -m onlyuav.train --help
```

## uv workflow (recommended)

The repo ships **`uv.lock`**, default **Aliyun PyPI** under `[[tool.uv.index]]`, and **`torch` from PyTorch CPU wheels** via `[tool.uv.sources]` (see `examples` extra).

```bash
cd /path/to/only-uav
uv sync --extra dev --extra examples
uv run python -m onlyuav.train --algo ppo --timesteps 4096 --model-path artifacts/uv_chain_ppo.zip --n-envs 2
uv run python -m onlyuav.eval --algo ppo --model-path artifacts/uv_chain_ppo.zip --n-episodes 2 --no-viz
uv run pytest -q
```

Paper-style **JSON + figures + `experiments.md`**: `uv run python -m onlyuav.experiments` (or `bash scripts/run_paper_experiment.sh`). Eval-only bundle: `uv run python -m onlyuav.eval ... --experiment-dir artifacts/my_run`.

## Optional install

```bash
pip install -e ".[dev,examples]"
```

Entry points: **`onlyuav-train`**, **`onlyuav-eval`** (need `[examples]` for SB3 + matplotlib).

## Layout

| Path | Role |
| --- | --- |
| [`onlyuav/models/channel`](onlyuav/models/channel) | Channel models |
| [`onlyuav/models/energy`](onlyuav/models/energy) | Energy models + ledger |
| [`onlyuav/environments`](onlyuav/environments) | Gymnasium envs |
| [`onlyuav/algorithms`](onlyuav/algorithms) | Named algorithms (**PPO**, **A2C** in `ppo.py` / `a2c.py`; SB3 as backend) |
| [`onlyuav/algorithms/registry.py`](onlyuav/algorithms/registry.py) | `train()` / `evaluate()` by `--algo` |
| [`onlyuav/metrics`](onlyuav/metrics) | Metrics + `EpisodeMetrics` |
| [`onlyuav/viz`](onlyuav/viz) | Step visualization, rendering |
| [`onlyuav/tools`](onlyuav/tools) | Small plotting helpers |
| [`onlyuav/train.py`](onlyuav/train.py) / [`onlyuav/eval.py`](onlyuav/eval.py) | CLI |

## Registered env IDs

After `import onlyuav`: `UAVGridOffload-v0`, `UAVContinuous2DOffload-v0`.

## Train / eval

```bash
python -m onlyuav.train --algo ppo --timesteps 20000
python -m onlyuav.train --algo a2c --timesteps 20000 --model-path artifacts/a2c_uav.zip
python -m onlyuav.train --timesteps 2000 --debug-viz
python -m onlyuav.eval --algo ppo --model-path artifacts/ppo_uav.zip --n-episodes 2
python -m onlyuav.eval --algo a2c --model-path artifacts/a2c_uav.zip --no-viz

Default checkpoint path is ``artifacts/<algo>_uav.zip`` when ``--model-path`` is omitted.

`onlyuav.eval` prints a short summary per episode (return, length) plus `task_completion_rate`, `coverage_rate`, `energy_total_j` when `episode_metrics` is present, then mean/std of returns.
```

Shell helpers (from repo root):

```bash
bash scripts/run_train.sh --timesteps 5000
bash scripts/run_eval.sh --model-path artifacts/ppo_uav.zip --no-viz
```

## Step visualization

See [`onlyuav/viz/wrapper.py`](onlyuav/viz/wrapper.py) and [`onlyuav/viz/step_visualizer.py`](onlyuav/viz/step_visualizer.py). UI strings are **English** so default matplotlib fonts render without CJK warnings. Normal training: omit `--debug-viz`. Eval: visualization on by default; use `--no-viz` on headless servers.

## Misc

- [`examples/custom_loop.py`](examples/custom_loop.py)  
- [`scripts/replay_recording.py`](scripts/replay_recording.py)  
- `from onlyuav.tools.charts import plot_scalar_series`

# onlyuav

基于文档《规划文档.md》复现的模块化无人机任务卸载仿真工程，核心特性：

- 组件注册 + 动态组装（`EnvBuilder`）
- 标准 `gymnasium.Env` 接口
- `Hydra` 多层配置（可按模块覆盖）
- `uv` 包管理与可选训练/评估流程（SB3）

## 快速开始（uv）

```bash
uv sync --extra dev
uv run python run_minimal.py
uv run pytest -q
```

## 源码直跑（无需安装本项目包）

你可以不执行 `pip install -e .`，直接按源码运行。`scripts/` 已内置路径引导：

```bash
# 仅需保证三方依赖已就绪（例如已 uv sync）
python scripts/train.py
python scripts/eval.py
```

`plot.py` 依赖 `matplotlib`，若需要画图请额外安装：

```bash
uv sync --extra examples
python scripts/plot.py --type train_curve --input logs/training_log.json --output plots/learning_curve.png
```

如果你从仓库外目录执行，也可以：

```bash
PYTHONPATH=/code/project/paper/uav/only-uav python /code/project/paper/uav/only-uav/scripts/train.py
```

## Hydra 多层配置

配置入口为 `configs/config.yaml`，通过 `defaults` 组合模块配置：

- `configs/modules/default.yaml`：基础模块拼装
- `configs/modules/*/*.yaml`：每个模块的具体实现配置
- `configs/experiment/*.yaml`：实验级别配置（训练步数、输出路径、对比实验）

示例：切换为“仅本地计算”实验

```bash
uv run python -m onlyuav.train experiment=local_only
uv run python -m onlyuav.eval experiment=local_only
```

示例：只覆盖某一个模块

```bash
uv run python -m onlyuav.train modules/computing=local_only
```

## 训练、评估、画图

```bash
uv run python -m onlyuav.train
uv run python -m onlyuav.eval
uv run python -m onlyuav.plot --type train_curve --input logs/training_log.json --output plots/learning_curve.png
uv run python -m onlyuav.plot --type eval_summary --input results/eval_metrics.json --output plots/eval_summary.png
```

## 工程结构

```text
onlyuav/
  core/            # 抽象接口、组件注册、环境构建
  models/          # mobility/channel/power/task/computing/...
  envs/            # DroneEnv
  evaluation/      # 指标与评估器
  train.py         # Hydra 训练入口
  eval.py          # Hydra 评估入口
configs/
  modules/         # 多层模块配置
  experiment/      # 实验配置
scripts/           # 与模块入口一致的脚本封装
tests/
```

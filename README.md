# onlyuav

基于文档《规划文档.md》复现的模块化无人机任务卸载仿真工程，核心特性：

- 组件注册 + 动态组装（`EnvBuilder`）
- 标准 `gymnasium.Env` 接口
- `Hydra` 多层配置（可按模块覆盖）
- `uv` 包管理与多 RL 库训练入口（SB3 / RLlib / Tianshou / DI-engine）

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
uv run python -m onlyuav.plot --type train_curve --input logs/training_log_ppo.json --output plots/learning_curve.png
uv run python -m onlyuav.plot --type eval_summary --input results/eval_metrics_ppo.json --output plots/eval_summary.png
```

## 多算法批量训练

通过 Hydra 列表参数一次训练多个算法。单次运行使用**同一个** `experiment.train.backend`（例如全部为 SB3 的 `ppo/a2c/sac/td3`，或全部为 RLlib 的 `ppo/sac`）：

```bash
python scripts/train.py experiment.train.algos=[ppo,sac,td3]
python scripts/eval.py experiment.eval.algos=[ppo,sac,td3]
```

`train.py` 在多算法模式下默认使用**多进程并行训练**（每个算法一个子进程）。如需改回串行：

```bash
python scripts/train.py experiment.train.algos=[ppo,sac,td3] experiment.train.parallel_multi_algo=false
```

输出会自动按算法分文件，例如：
- `trained_models/ppo_edge.zip`
- `trained_models/sac_edge.zip`
- `results/eval_metrics_ppo.json`
- `results/eval_metrics_sac.json`

## 算法切换与默认超参（可行性说明）

**同一开源库内「只改算法名、其它参数完全不动」是否可行？**

- **部分可行**：像 `learning_rate`、`gamma` 这类多数算法都有的量，可以抽到 `experiment.train.shared_algo_kwargs` 里统一写；但 PPO 的 `n_steps`、SAC 的 `buffer_size`、TD3 的 `policy_delay` 等**算法专有**项无法在「一套扁平超参」里做到语义完全一致，否则要么无效键、要么库报错。
- **本仓库做法**：在 `onlyuav/rl/builtin_defaults.py` 为每个 `(backend, algorithm)` 维护**已对齐该库 API 的默认字典**；你只改 `experiment.train.algorithm`（或列表 `experiment.train.algos`）即可切换，**不必**为每个算法单独写 YAML。
- **叠加优先级（从低到高）**：内置默认 → `shared_algo_kwargs` → 若存在 `configs/algorithm/<backend>/` 下与算法同名的 YAML 则合并 →（可选）用 Hydra 组 `algorithm/sb3` 覆盖当前 SB3 算法（配置树为 `algorithm.sb3.*`）→ `experiment.train.algo_kwargs`。

| backend | 依赖 | 内置算法 | 可选 YAML 目录（`configs/algorithm/…`） |
|---------|------|-----------|----------------|
| `sb3` | 默认 | ppo, a2c, sac, td3 | `sb3/` |
| `rllib` | `--extra rllib` | ppo, a2c, sac, td3, ddpg | `rlib/` |
| `tianshou` | `--extra tianshou` | ppo, a2c, sac, td3, ddpg | `tianshou/` |
| `di_engine` | `--extra di-engine` | ppo（训练入口仅对接 PPO） | `ding/` |

说明：`rllib` 的 **td3 / ddpg / a2c** 依赖旧版 Ray 仍提供的 `*Config`；若当前环境导入失败，请改用 **sb3** 或 **tianshou**，或为 RLlib 安装仍包含这些算法的 Ray 版本。

示例（只改算法名）：

```bash
uv run python -m onlyuav.train experiment.train.algorithm=sac
uv run python -m onlyuav.train experiment.train.algorithm=td3 experiment.train.backend=sb3
```

多算法并行（SB3 四种一次跑完）：

```bash
uv run python -m onlyuav.train 'experiment.train.algos=[ppo,a2c,sac,td3]' experiment.train.parallel_multi_algo=true
```

后台跑、日志进 `output.log`：

```bash
bash scripts/nohup_train_all_sb3.sh 50000   # 第二个参数为每算法步数，默认 50000
# 或: nohup uv run python -m onlyuav.train ... >> output.log 2>&1 &
```

后台评估四算法（与上面 `algos` 一致），日志进 `output_eval.log`：

```bash
bash scripts/nohup_eval_all_sb3.sh 30   # 参数为每算法评估 episode 数，默认 30
```

若要用 Hydra 覆盖 SB3 超参，请在 `defaults` 中加入如 `- algorithm/sb3: ppo`（与 `configs/algorithm/sb3/ppo.yaml` 对应），或命令行 `+algorithm/sb3=ppo`；合并进 `load_algo_hparams` 的键来自配置树中的 **`algorithm.sb3`**。

`experiment.train.algo_kwargs` 仍为**最高优先级**覆盖。

RLlib 专有项见 `experiment.train.rllib`。

**评估**：`onlyuav.eval` 仅支持 SB3 的 `.zip`；其它 backend 需在各库内评估。

## 断点续训（默认开启）

训练默认会自动检查模型文件并续训：
- 若存在 checkpoint：日志标记为 `RESUME`
- 若不存在 checkpoint：日志标记为 `FRESH_START`

可显式关闭续训：

```bash
python scripts/train.py experiment.train.resume=false
```

如果希望每轮都复用同一批场景内容（固定 reset seed），保持：

```bash
python scripts/train.py experiment.train.fixed_reset_seed=42
```

若要恢复随机刷新场景：

```bash
python scripts/train.py experiment.train.fixed_reset_seed=null
```

训练新增日志文件：
- 执行日志：`outputs/<date>/<time>/train.log`（Hydra 单一任务日志）
- 每轮指标：`outputs/<date>/<time>/<metric_jsonl>`（默认 `logs/train_metrics_{algo}.jsonl`）

推荐指标设置（默认）：
- `experiment.train.metrics_log_every_steps=100`
- `experiment.train.metrics_include_step_reward=false`（减少噪声）
- `experiment.train.metrics_include_timestamp=true`

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

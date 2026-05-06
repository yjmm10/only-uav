#!/usr/bin/env bash
# 后台训练 SB3 支持的全部算法（ppo/a2c/sac/td3），日志写入项目根目录 output.log
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

STEPS="${1:-20000}"

nohup uv run python -m onlyuav.train \
  experiment.train.backend=sb3 \
  'experiment.train.algos=[ppo,a2c,sac,td3]' \
  "experiment.train.total_timesteps=${STEPS}" \
  experiment.train.parallel_multi_algo=true \
  >> output.log 2>&1 &

echo "PID=$! 日志: $ROOT/output.log 步数/算法=${STEPS}"

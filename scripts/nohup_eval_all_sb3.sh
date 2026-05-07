#!/usr/bin/env bash
# 后台评估 SB3 下已训练的多算法（ppo/a2c/sac/td3），日志写入项目根目录 output_eval.log
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

EPISODES="${1:-30}"

nohup uv run python -m onlyuav.eval \
  experiment.train.backend=sb3 \
  'experiment.eval.algos=[ppo,a2c,sac,td3]' \
  "experiment.eval.episodes=${EPISODES}" \
  >> output_eval.log 2>&1 &

echo "PID=$! 日志: $ROOT/output_eval.log episodes=${EPISODES}"

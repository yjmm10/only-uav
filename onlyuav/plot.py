from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(log_path: Path, save_path: Path, window: int = 5) -> None:
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    timesteps = data["timesteps"]
    rewards = data["episode_rewards"]
    if len(rewards) < max(window, 1):
        smoothed = rewards
        ts_smoothed = timesteps
    else:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ts_smoothed = timesteps[window - 1 :]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(ts_smoothed, smoothed)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward (smoothed)")
    plt.title("Training Learning Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curve saved: {save_path}")


def plot_eval_summary(summary_path: Path, save_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = ["avg_reward", "avg_completed_tasks", "avg_delay", "avg_energy", "avg_offload_rate"]
    labels = ["Reward", "Completed", "Delay", "Energy", "OffloadRate"]
    means = [summary[m]["mean"] for m in metrics if m in summary]
    stds = [summary[m]["std"] for m in metrics if m in summary]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(means))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation summary saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["train_curve", "eval_summary"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="plot.png")
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    if args.type == "train_curve":
        plot_learning_curve(Path(args.input), Path(args.output), args.window)
    else:
        plot_eval_summary(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

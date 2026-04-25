from __future__ import annotations

from typing import Any

import numpy as np


class MetricsTracker:
    """记录逐步信息，并在 episode 结束时汇总指标。"""

    def __init__(self):
        self.reset_session()

    def reset_session(self):
        self.episode_data = []
        self.current_episode = None

    def begin_episode(self):
        self.current_episode = {
            "reward": [],
            "task_completed": 0,
            "task_failed": 0,
            "delay_list": [],
            "energy_consumed": 0.0,
            "offload_actions": [],
            "distance_list": [],
        }

    def record_step(self, action, reward, info):
        if self.current_episode is None:
            return
        self.current_episode["reward"].append(float(reward))
        self.current_episode["offload_actions"].append(int(info.get("offload_target", 0)))
        self.current_episode["delay_list"].append(float(info.get("delay", 0.0)))
        self.current_episode["distance_list"].append(float(info.get("distance", 0.0)))
        self.current_episode["task_completed"] += int(info.get("completed", 0))
        self.current_episode["task_failed"] += int(info.get("failed", 0))

    def end_episode(self, final_info):
        if self.current_episode is None:
            return
        ep = self.current_episode
        ep["total_reward"] = float(np.sum(ep["reward"]))
        ep["avg_delay"] = float(np.mean(ep["delay_list"])) if ep["delay_list"] else 0.0
        ep["energy_consumed"] = float(final_info.get("total_energy", 0.0))
        ep["offload_rate"] = float(np.mean(ep["offload_actions"])) if ep["offload_actions"] else 0.0
        ep["avg_distance"] = float(np.mean(ep["distance_list"])) if ep["distance_list"] else 0.0
        total_task = ep["task_completed"] + ep["task_failed"]
        ep["completion_rate"] = float(ep["task_completed"] / total_task) if total_task > 0 else 0.0
        self.episode_data.append(ep)
        self.current_episode = None

    def get_summary(self) -> dict[str, Any]:
        if not self.episode_data:
            return {}
        rewards = [ep["total_reward"] for ep in self.episode_data]
        completions = [ep["task_completed"] for ep in self.episode_data]
        delays = [ep["avg_delay"] for ep in self.episode_data]
        energies = [ep["energy_consumed"] for ep in self.episode_data]
        offloads = [ep["offload_rate"] for ep in self.episode_data]
        distances = [ep["avg_distance"] for ep in self.episode_data]
        return {
            "num_episodes": len(self.episode_data),
            "avg_reward": (float(np.mean(rewards)), float(np.std(rewards))),
            "avg_completed_tasks": (float(np.mean(completions)), float(np.std(completions))),
            "avg_delay": (float(np.mean(delays)), float(np.std(delays))),
            "avg_energy": (float(np.mean(energies)), float(np.std(energies))),
            "avg_offload_rate": (float(np.mean(offloads)), float(np.std(offloads))),
            "avg_distance": (float(np.mean(distances)), float(np.std(distances))),
        }

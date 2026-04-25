from __future__ import annotations

from typing import Any, Callable

from onlyuav.evaluation.metrics import MetricsTracker


def evaluate_policy(env, policy_fn: Callable[[Any], Any], num_episodes: int = 10, verbose: bool = True):
    tracker = MetricsTracker()
    for idx in range(num_episodes):
        obs, _ = env.reset()
        tracker.begin_episode()
        done = False
        info = {}
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.record_step(action, reward, info)
            done = terminated or truncated
        tracker.end_episode(info)
        if verbose:
            ep = tracker.episode_data[-1]
            print(f"Episode {idx + 1}/{num_episodes}: reward={ep['total_reward']:.2f}, completed={ep['task_completed']}")
    return tracker.get_summary(), tracker

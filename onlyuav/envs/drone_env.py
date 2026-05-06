import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        mobility,
        channel,
        power,
        task_gen,
        computing,
        energy,
        reward_model,
        obs_model,
        action_interp,
        server_pos,
        max_steps=200,
        dt=1.0,
    ):
        super().__init__()
        self.mobility = mobility
        self.channel = channel
        self.power = power
        self.task_gen = task_gen
        self.computing = computing
        self.energy_model = energy
        self.reward_model = reward_model
        self.obs_model = obs_model
        self.action_interp = action_interp
        self.server_pos = np.array(server_pos, dtype=np.float64)
        self.max_steps = max_steps
        self.dt = dt

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.task_queue = []
        self.steps = 0
        self.total_completed = 0
        self.total_delay = 0.0
        self.total_energy = 0.0
        self.last_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.task_queue = []
        self.total_completed = 0
        self.total_delay = 0.0
        self.total_energy = 0.0
        self.last_info = {}
        self.mobility.reset()
        self.energy_model.reset()
        self.task_gen.reset()
        self.computing.reset()
        self.action_interp.reset()
        self.reward_model.reset()
        return self._get_obs(), {}

    def step(self, action):
        move_cmd, offload_target, cpu_freq = self.action_interp.interpret(action)

        # 移动与信道
        state = self.mobility.step(move_cmd)
        pos, vel = state["pos"], state["vel"]
        rate = self.channel.rate(pos, self.server_pos)

        # 任务到达
        self.task_queue.extend(self.task_gen.sample(self.steps))

        # 处理任务（简化为每步最多处理一个）
        delay = 0.0
        comp_energy = 0.0
        completed = 0
        failed = 0
        if self.task_queue:
            task = self.task_queue.pop(0)
            result = self.computing.process(task, offload_target, rate)
            success = result["success"] and result["exec_time"] <= task["max_delay"]
            delay = float(result["exec_time"] if success else task["max_delay"])
            comp_energy = float(result["energy"])
            completed = int(success)
            failed = int(not success)
            self.total_completed += completed

        # 功耗与电量
        power_total = self.power.compute(vel, cpu_freq)
        self.energy_model.consume(power_total, dt=self.dt)
        self.total_delay += delay
        self.total_energy += power_total + comp_energy

        # 奖励
        reward = self.reward_model.compute(
            {"completed": completed, "energy_cost": power_total + comp_energy, "delay": delay}
        )

        self.steps += 1
        terminated = self.energy_model.remaining() <= 0.0
        truncated = self.steps >= self.max_steps
        distance = float(np.linalg.norm(pos[:2] - self.server_pos[:2]))
        info = {
            "completed": completed,
            "failed": failed,
            "delay": delay,
            "offload_target": offload_target,
            "distance": distance,
            "total_energy": self.total_energy,
            "remaining_energy": self.energy_model.remaining(),
            "completed_tasks": self.total_completed,
        }
        self.last_info = info
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        state = self.mobility.state()
        env_state = {
            "pos": state["pos"],
            "vel": state["vel"],
            "energy": self.energy_model.remaining(),
            "queue_len": len(self.task_queue),
            "server_pos": self.server_pos,
        }
        return self.obs_model.get_obs(env_state)

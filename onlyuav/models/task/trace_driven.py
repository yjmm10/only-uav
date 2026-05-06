"""轨迹驱动任务到达：按预设行顺序每步最多释放一个任务，可选从 JSON 文件加载。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import ITaskGenerator


def _plain_rows(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if OmegaConf.is_config(rows):
        return OmegaConf.to_container(rows, resolve=True)  # type: ignore[return-value]
    return list(rows)


@ComponentRegistry.register("TraceDriven")
class TraceDrivenTasks(ITaskGenerator):
    def __init__(
        self,
        trace_file: str | None = None,
        rows: Any | None = None,
        max_delay: float = 2.0,
        loop: bool = True,
    ):
        self.max_delay = max_delay
        self.loop = bool(loop)
        self._rows: list[dict[str, Any]] = []
        if trace_file:
            path = Path(trace_file)
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._rows = raw
            else:
                self._rows = raw.get("tasks", [])
        else:
            self._rows = _plain_rows(rows)
        if not self._rows:
            self._rows = [{"data_size": 1e6, "req_cycles": 5e8}]
        self._idx = 0
        self.next_id = 0

    def reset(self, **kwargs):
        self._idx = 0
        self.next_id = 0

    def sample(self, current_time):
        if self._idx >= len(self._rows):
            if not self.loop:
                return []
            self._idx = 0
        rec = dict(self._rows[self._idx])
        self._idx += 1
        task = {
            "id": self.next_id,
            "data_size": float(rec.get("data_size", 1e6)),
            "req_cycles": float(rec.get("req_cycles", 5e8)),
            "max_delay": float(rec.get("max_delay", self.max_delay)),
            "arrival_time": current_time,
        }
        self.next_id += 1
        return [task]

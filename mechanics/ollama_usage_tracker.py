"""Shared utilities for capturing Ollama token and latency metrics."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict


def _default_state() -> Dict[str, Any]:
    return {
        "started_at": time.time(),
        "embedding": {
            "calls": 0,
            "prompt_tokens": 0,
            "eval_tokens": 0,
            "prompt_duration_ns": 0,
            "eval_duration_ns": 0,
            "total_duration_ns": 0,
            "wall_time_s": 0.0,
        },
        "generation": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prompt_duration_ns": 0,
            "eval_duration_ns": 0,
            "total_duration_ns": 0,
            "wall_time_s": 0.0,
        },
        "total_wall_time_s": 0.0,
    }


class OllamaUsageTracker:
    """Lightweight collector for Ollama embedding and generation usage."""

    def __init__(self) -> None:
        self._state = _default_state()

    def reset(self) -> None:
        """Reset the tracker for a new query."""
        self._state = _default_state()

    def record_embedding(
        self,
        *,
        prompt_tokens: int = 0,
        eval_tokens: int = 0,
        prompt_duration_ns: int = 0,
        eval_duration_ns: int = 0,
        total_duration_ns: int = 0,
        wall_time_s: float = 0.0,
    ) -> None:
        """Accumulate stats for an embedding call."""
        data = self._state["embedding"]
        data["calls"] += 1
        data["prompt_tokens"] += max(prompt_tokens, 0)
        data["eval_tokens"] += max(eval_tokens, 0)
        data["prompt_duration_ns"] += max(prompt_duration_ns, 0)
        data["eval_duration_ns"] += max(eval_duration_ns, 0)
        data["total_duration_ns"] += max(total_duration_ns, 0)
        data["wall_time_s"] += max(wall_time_s, 0.0)

    def record_generation(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_duration_ns: int = 0,
        eval_duration_ns: int = 0,
        total_duration_ns: int = 0,
        wall_time_s: float = 0.0,
    ) -> None:
        """Accumulate stats for a chat/generation call."""
        data = self._state["generation"]
        data["prompt_tokens"] += max(prompt_tokens, 0)
        data["completion_tokens"] += max(completion_tokens, 0)
        data["prompt_duration_ns"] += max(prompt_duration_ns, 0)
        data["eval_duration_ns"] += max(eval_duration_ns, 0)
        data["total_duration_ns"] += max(total_duration_ns, 0)
        data["wall_time_s"] += max(wall_time_s, 0.0)

    def finalize(self, total_wall_time_s: float) -> None:
        """Store the end-to-end wall time for the current query."""
        self._state["total_wall_time_s"] = max(total_wall_time_s, 0.0)

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current metrics."""
        return copy.deepcopy(self._state)


usage_tracker = OllamaUsageTracker()

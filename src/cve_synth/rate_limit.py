from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic


@dataclass(slots=True)
class APIKeyState:
    api_key: str
    min_interval_seconds: float
    cooldown_seconds: float = 0.0
    available_at: float = 0.0
    last_used_at: float = 0.0
    failure_count: int = 0
    uses: int = 0
    disabled: bool = False


class MultiKeyRateLimiter:
    def __init__(self, api_keys: list[str], *, min_interval_seconds: float = 1.0, cooldown_seconds: float = 30.0) -> None:
        if not api_keys:
            raise ValueError("api_keys must not be empty")
        self._lock = Lock()
        self._states = [
            APIKeyState(api_key=key.strip(), min_interval_seconds=min_interval_seconds, cooldown_seconds=cooldown_seconds)
            for key in api_keys
            if key.strip()
        ]
        if not self._states:
            raise ValueError("api_keys must contain at least one non-empty key")
        self._cursor = 0

    def acquire(self) -> str:
        with self._lock:
            now = monotonic()
            for offset in range(len(self._states)):
                index = (self._cursor + offset) % len(self._states)
                state = self._states[index]
                if state.disabled:
                    continue
                if state.available_at <= now:
                    state.last_used_at = now
                    state.uses += 1
                    state.available_at = now + state.min_interval_seconds
                    self._cursor = (index + 1) % len(self._states)
                    return state.api_key
        raise RuntimeError("No API key is currently available; all keys are rate-limited or disabled")

    def mark_success(self, api_key: str) -> None:
        with self._lock:
            state = self._find(api_key)
            state.failure_count = 0

    def mark_rate_limited(self, api_key: str, retry_after_seconds: float | None = None) -> None:
        with self._lock:
            state = self._find(api_key)
            state.failure_count += 1
            backoff = retry_after_seconds if retry_after_seconds and retry_after_seconds > 0 else state.cooldown_seconds
            state.available_at = max(state.available_at, monotonic() + backoff)

    def mark_failure(self, api_key: str, *, disable_after: int = 6) -> None:
        with self._lock:
            state = self._find(api_key)
            state.failure_count += 1
            if state.failure_count >= disable_after:
                state.disabled = True
            state.available_at = monotonic() + state.cooldown_seconds

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "api_key": state.api_key,
                    "available_at": state.available_at,
                    "last_used_at": state.last_used_at,
                    "failure_count": state.failure_count,
                    "uses": state.uses,
                    "disabled": state.disabled,
                }
                for state in self._states
            ]

    def _find(self, api_key: str) -> APIKeyState:
        for state in self._states:
            if state.api_key == api_key:
                return state
        raise KeyError(f"Unknown api_key: {api_key}")

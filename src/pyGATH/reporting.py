"""Simulation-scoped progress reporting for interactive and file-backed runs."""

from __future__ import annotations

import contextvars
import functools
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TextIO


def format_duration(seconds: float | None) -> str:
    """Format a nonnegative duration as ``HH:MM:SS`` or ``HH:MM:SS.mmm``."""
    if seconds is None:
        return "unknown"
    milliseconds = max(0, round(float(seconds) * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


@dataclass(frozen=True)
class ReportingConfig:
    """Validated output controls loaded from an input deck."""

    verbosity: int = 0
    console: bool = True
    file: Path | None = None
    progress_interval_s: float = 5.0

    def reporter(self) -> SimulationReporter:
        """Create a fresh reporter whose clock starts on context entry."""
        return SimulationReporter(self)


class SimulationReporter:
    """Route timestamped simulation events to the console and an optional file."""

    def __init__(self, config: ReportingConfig, *, stream: TextIO | None = None):
        self.config = config
        self._stream = stream
        self._file_stream: TextIO | None = None
        self._started = 0.0
        self._token: contextvars.Token[SimulationReporter | None] | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.config.verbosity > 0 and (
            self.config.console or self.config.file is not None
        )

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started if self._started else 0.0

    def __enter__(self) -> Self:
        if self._token is not None:
            raise RuntimeError("a SimulationReporter cannot be entered twice")
        self._started = time.perf_counter()
        if self.enabled and self.config.file is not None:
            self.config.file.parent.mkdir(parents=True, exist_ok=True)
            self._file_stream = self.config.file.open("w", encoding="utf-8")
        self._token = _ACTIVE_REPORTER.set(self)
        self.emit(1, "simulation", "simulation started")
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        try:
            if exception is None:
                self.emit(1, "simulation", "simulation complete")
            else:
                self.emit(
                    1,
                    "simulation",
                    f"simulation failed: {exception_type.__name__}: {exception}",
                )
        finally:
            if self._token is not None:
                _ACTIVE_REPORTER.reset(self._token)
                self._token = None
            if self._file_stream is not None:
                self._file_stream.close()
                self._file_stream = None
        return False

    def emit(self, level: int, stage: str, message: str) -> None:
        """Emit an event when the configured verbosity includes ``level``."""
        if not self.enabled or self.config.verbosity < level:
            return
        line = f"[+{format_duration(self.elapsed_s)}] {stage}: {message}"
        with self._lock:
            if self.config.console:
                stream = self._stream if self._stream is not None else sys.stdout
                print(line, file=stream, flush=True)
            if self._file_stream is not None:
                print(line, file=self._file_stream, flush=True)

    def callback(self, event: dict[str, Any]) -> None:
        """Consume a structured kernel progress event."""
        status = str(event.get("status", "running"))
        level = 1 if status in {"failed", "error"} else 2
        message = str(event.get("message", status))
        eta = event.get("estimated_remaining_s")
        if eta is not None:
            message += f"; ETA {format_duration(float(eta))}"
        self.emit(level, str(event.get("stage", "progress")), message)
        details = {
            key: value
            for key, value in event.items()
            if key not in {"stage", "status", "message", "estimated_remaining_s"}
        }
        if details:
            rendered = ", ".join(f"{key}={value}" for key, value in details.items())
            self.emit(3, f"{event.get('stage', 'progress')} details", rendered)


_ACTIVE_REPORTER: contextvars.ContextVar[SimulationReporter | None] = (
    contextvars.ContextVar("pyGATH_active_reporter", default=None)
)


def get_reporter() -> SimulationReporter | None:
    """Return the reporter active in the current thread or async context."""
    return _ACTIVE_REPORTER.get()


def emit(level: int, stage: str, message: str) -> None:
    """Emit through the active reporter, if any."""
    reporter = get_reporter()
    if reporter is not None:
        reporter.emit(level, stage, message)


def synchronize(value: Any) -> None:
    """Wait for JAX leaves so reported stage durations include device execution."""
    try:
        import jax
    except ImportError:  # pragma: no cover - JAX is a required dependency
        return
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def reported_stage(
    name: str,
    *,
    describe: Callable[[Any], str] | None = None,
    synchronize_result: bool = True,
):
    """Decorate a public workflow boundary with start/completion reporting."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            reporter = get_reporter()
            if reporter is None or not reporter.enabled:
                return function(*args, **kwargs)
            reporter.emit(1, name, "started")
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                if synchronize_result:
                    synchronize(result)
            except Exception as error:
                elapsed = time.perf_counter() - started
                reporter.emit(
                    1,
                    name,
                    f"failed after {format_duration(elapsed)}: "
                    f"{type(error).__name__}: {error}",
                )
                raise
            elapsed = time.perf_counter() - started
            detail = f"; {describe(result)}" if describe is not None else ""
            reporter.emit(
                1,
                name,
                f"complete in {format_duration(elapsed)}{detail}",
            )
            return result

        return wrapped

    return decorate


class ProgressTracker:
    """Throttled numerical progress and ETA reporting for host-controlled loops."""

    def __init__(self, stage: str, total: int, unit: str):
        self.reporter = get_reporter()
        self.stage = stage
        self.total = int(total)
        self.unit = unit
        self.started = time.perf_counter()
        self.last_reported = self.started

    def update(
        self,
        completed: int,
        *,
        message: str | None = None,
        force: bool = False,
    ) -> None:
        reporter = self.reporter
        if reporter is None or not reporter.enabled or reporter.config.verbosity < 2:
            return
        now = time.perf_counter()
        completed = int(completed)
        if not (
            force
            or completed >= self.total
            or now - self.last_reported >= reporter.config.progress_interval_s
        ):
            return
        elapsed = now - self.started
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        eta = (self.total - completed) / rate if rate > 0.0 else None
        prefix = f"{message}; " if message else ""
        reporter.emit(
            2,
            self.stage,
            f"{prefix}{completed:,}/{self.total:,} {self.unit}; "
            f"ETA {format_duration(eta)}",
        )
        self.last_reported = now


__all__ = [
    "ProgressTracker",
    "ReportingConfig",
    "SimulationReporter",
    "emit",
    "format_duration",
    "get_reporter",
    "reported_stage",
    "synchronize",
]

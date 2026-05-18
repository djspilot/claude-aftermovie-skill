"""One home for the "try-import → warn-once → fall back" pattern.

Every analyzer that depends on a third-party Python package (cv2, mediapipe,
Pillow, pillow-heif) or an external CLI (exiftool, ffmpeg/ffprobe) used to
reinvent the same dance: try the import, stash a module-level `_WARNED` flag,
short-circuit on missing, log once on the first miss. Four near-identical
copies, one per analyzer; the shared shape was invisible and `cmd_doctor`
had to mirror each `import foo` to ask "is `foo` here?".

This module owns the answer. Callers get:

  * `OptionalImport` — a wrapped `import name`. `.available` answers the
    predicate, `.module` exposes the module (or None), `.require()` returns
    the module and logs ONCE on the first call when missing. `bool(dep)` is
    a synonym for `.available` so the value can be used directly as a guard.

  * `OptionalCommand` — same shape, but for executables on PATH (exiftool,
    ffmpeg). `.path` carries the resolved path; `.require()` returns it or
    None and warns-once on the first miss.

Instances are cached per name so two callers asking for the same dep share
the same one-shot warning state — there is no way to spam the log per clip
anymore, regardless of how many analyzers consume the dep.

`cmd_doctor` queries this module instead of re-doing `__import__("foo")` so
the truth-of-availability has one home.
"""
from __future__ import annotations

import importlib
import shutil
from typing import Any

from aftermovie.ffmpeg_cmd import log


class OptionalImport:
    """A lazily-resolved optional Python module with one-shot warning state.

    Invariants:
        - The underlying `importlib.import_module(name)` runs exactly once at
          construction. `.available` reflects that single attempt.
        - `.require()` logs the configured `warning` exactly once across the
          lifetime of the instance, regardless of how many callers hit the
          missing-dep path. Subsequent misses are silent.
        - `bool(self)` == `self.available`, so callers can write
          `if dep:` as a predicate.
    """

    __slots__ = ("name", "warning", "module", "_warned")

    def __init__(self, name: str, *, warning: str) -> None:
        self.name = name
        self.warning = warning
        self._warned = False
        try:
            self.module: Any | None = importlib.import_module(name)
        except ImportError:
            self.module = None

    @property
    def available(self) -> bool:
        return self.module is not None

    def __bool__(self) -> bool:
        return self.available

    def require(self) -> Any | None:
        """Return the module, or None after logging `warning` once."""
        if self.module is not None:
            return self.module
        if not self._warned:
            log(self.warning)
            self._warned = True
        return None

    def reset_warning(self) -> None:
        """Re-arm the one-shot warning. Tests only."""
        self._warned = False


class OptionalCommand:
    """A PATH-resolved optional CLI with the same shape as OptionalImport.

    Invariants:
        - `shutil.which(name)` runs at construction; the result is the value
          of `.path` (None when missing).
        - `.require()` logs `warning` exactly once across the lifetime of
          the instance when the command is absent.
        - `bool(self)` == `self.available`.
    """

    __slots__ = ("name", "warning", "path", "_warned")

    def __init__(self, name: str, *, warning: str) -> None:
        self.name = name
        self.warning = warning
        self._warned = False
        self.path: str | None = shutil.which(name)

    @property
    def available(self) -> bool:
        return self.path is not None

    def __bool__(self) -> bool:
        return self.available

    def require(self) -> str | None:
        if self.path is not None:
            return self.path
        if not self._warned:
            log(self.warning)
            self._warned = True
        return None

    def reset_warning(self) -> None:
        """Re-arm the one-shot warning. Tests only."""
        self._warned = False


# Cache of (kind, name) → instance so two analyzers asking for the same dep
# share one warn-once state. Wiped in tests via _reset_for_tests().
_REGISTRY: dict[tuple[str, str], Any] = {}


def optional_import(name: str, *, warning: str) -> OptionalImport:
    """Get-or-create the OptionalImport for `name`.

    Two callers asking for the same module get the same instance — and thus
    the same one-shot warning state. The `warning` of the first caller wins;
    later callers' warnings are ignored (no analyzer should be racing to
    rename a dep's message, and silently re-binding it would be surprising).
    """
    key = ("import", name)
    dep = _REGISTRY.get(key)
    if dep is None:
        dep = OptionalImport(name, warning=warning)
        _REGISTRY[key] = dep
    return dep


def optional_command(name: str, *, warning: str) -> OptionalCommand:
    """Get-or-create the OptionalCommand for `name`. See `optional_import`."""
    key = ("command", name)
    dep = _REGISTRY.get(key)
    if dep is None:
        dep = OptionalCommand(name, warning=warning)
        _REGISTRY[key] = dep
    return dep


def _reset_for_tests() -> None:
    """Clear the registry. Tests only — do not call from production code."""
    _REGISTRY.clear()

"""In-process job runner — long-running ops return a job id and poll for status."""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from typing import Any, Callable

from aftermovie.state import load_job, save_job

# active threads kept in-process so cancel_job can join() if needed
_threads: dict[str, threading.Thread] = {}
_cancel_flags: dict[str, threading.Event] = {}


def _now() -> float:
    return time.time()


def start_job(kind: str, fn: Callable[[threading.Event], dict[str, Any]]) -> str:
    """Run `fn` in a background thread. fn gets a cancel Event; it can return
    any JSON-serializable result. Returns the job id."""
    job_id = secrets.token_hex(6)
    cancel = threading.Event()
    _cancel_flags[job_id] = cancel
    save_job(job_id, {
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "started_at": _now(),
        "result": None,
        "error": None,
    })

    def _runner() -> None:
        try:
            result = fn(cancel)
            job = load_job(job_id)
            job["status"] = "cancelled" if cancel.is_set() else "done"
            job["finished_at"] = _now()
            job["result"] = result
            save_job(job_id, job)
        except Exception as e:
            job = load_job(job_id)
            job["status"] = "error"
            job["finished_at"] = _now()
            job["error"] = f"{type(e).__name__}: {e}"
            job["traceback"] = traceback.format_exc()
            save_job(job_id, job)

    t = threading.Thread(target=_runner, daemon=True, name=f"job-{kind}-{job_id}")
    _threads[job_id] = t
    t.start()
    return job_id


def get_status(job_id: str) -> dict[str, Any]:
    return load_job(job_id)


def cancel(job_id: str) -> bool:
    flag = _cancel_flags.get(job_id)
    if flag is None:
        return False
    flag.set()
    return True


def wait(job_id: str, timeout: float | None = None) -> dict[str, Any]:
    """Block until the job leaves the `running` state, then return its record."""
    t = _threads.get(job_id)
    if t is not None:
        t.join(timeout=timeout)
    return load_job(job_id)

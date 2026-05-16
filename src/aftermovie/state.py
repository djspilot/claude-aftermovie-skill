"""On-disk state for catalogs, plans, and render jobs.

Layout:
    ~/.skills-data/aftermovie/
        catalogs/<catalog_id>.json
        plans/<plan_id>.json
        renders/<job_id>.json

IDs are short content hashes so the same input always produces the same id.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aftermovie.config import data_dir


def _dir(name: str) -> Path:
    d = data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def catalog_dir() -> Path:  return _dir("catalogs")
def plan_dir() -> Path:     return _dir("plans")
def render_dir() -> Path:   return _dir("renders")


# ---- IDs --------------------------------------------------------------------

def _hash(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def catalog_id_for(folder: Path) -> str:
    """Stable id derived from folder path + sorted file list + mtimes."""
    folder = folder.resolve()
    pieces = [str(folder)]
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            try:
                st = f.stat()
                pieces.append(f"{f.relative_to(folder)}|{st.st_size}|{int(st.st_mtime)}")
            except OSError:
                continue
    return _hash("\n".join(pieces))


def plan_id_for(catalog_id: str, song_path: Path, theme: str | None,
                target_length_s: float | None, aspect: str, seed: int) -> str:
    parts = [
        catalog_id,
        str(Path(song_path).resolve()),
        theme or "",
        f"{target_length_s or ''}",
        aspect,
        str(seed),
    ]
    return _hash("|".join(parts))


# ---- I/O --------------------------------------------------------------------

def save_catalog(catalog_id: str, catalog: dict[str, Any]) -> Path:
    p = catalog_dir() / f"{catalog_id}.json"
    p.write_text(json.dumps(catalog, indent=2))
    return p


def load_catalog(catalog_id: str) -> dict[str, Any]:
    p = catalog_dir() / f"{catalog_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"catalog {catalog_id} not found")
    return json.loads(p.read_text())


def save_plan(plan_id: str, plan: dict[str, Any]) -> Path:
    p = plan_dir() / f"{plan_id}.json"
    p.write_text(json.dumps(plan, indent=2))
    return p


def load_plan(plan_id: str) -> dict[str, Any]:
    p = plan_dir() / f"{plan_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"plan {plan_id} not found")
    return json.loads(p.read_text())


def save_job(job_id: str, job: dict[str, Any]) -> Path:
    p = render_dir() / f"{job_id}.json"
    p.write_text(json.dumps(job, indent=2))
    return p


def load_job(job_id: str) -> dict[str, Any]:
    p = render_dir() / f"{job_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"job {job_id} not found")
    return json.loads(p.read_text())

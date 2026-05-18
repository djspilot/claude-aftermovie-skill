"""On-disk state — thin shim over `aftermovie.repos`.

Historically this module owned the path constructors, id derivation, and
JSON I/O for catalogs / plans / render jobs. The catalog + plan semantics
have moved into `aftermovie.repos` (`CatalogRepository`, `PlanRepository`)
so callers can ask in domain terms instead of reaching at file paths.

The functions below are kept as a back-compat surface — they delegate to
the repository singletons. New code should import `repos` directly.

Render-job persistence still lives here; jobs are simple enough (one dict
per job, fetched by id) that they don't need a repository abstraction.

Layout:
    ~/.skills-data/aftermovie/
        catalogs/<catalog_id>.json    -- owned by CatalogRepository
        plans/<plan_id>.json          -- owned by PlanRepository
        renders/<job_id>.json         -- owned here (load_job / save_job)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aftermovie.config import data_dir
from aftermovie.repos import (
    CatalogRepository,
    PlanRepository,
    catalog_repo,
    plan_repo,
)

__all__ = [
    "catalog_dir", "plan_dir", "render_dir",
    "catalog_id_for", "plan_id_for",
    "save_catalog", "load_catalog",
    "save_plan", "load_plan",
    "save_job", "load_job",
    "data_dir",
    "CatalogRepository", "PlanRepository",
    "catalog_repo", "plan_repo",
]


# ---- dir helpers (kept so monkeypatch / external callers don't break) -----

def catalog_dir() -> Path:
    d = data_dir() / "catalogs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plan_dir() -> Path:
    d = data_dir() / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_dir() -> Path:
    d = data_dir() / "renders"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- id derivation (delegates to repos) -----------------------------------

def catalog_id_for(folder: Path) -> str:
    return catalog_repo.id_for(folder)


def plan_id_for(catalog_id: str, song_path: Path, theme: str | None,
                target_length_s: float | None, aspect: str, seed: int) -> str:
    return plan_repo.id_for(catalog_id, song_path, theme,
                            target_length_s, aspect, seed)


# ---- catalog + plan I/O (delegates to repos) ------------------------------

def save_catalog(catalog_id: str, catalog: dict[str, Any]) -> Path:
    return catalog_repo.save_raw(catalog_id, catalog)


def load_catalog(catalog_id: str) -> dict[str, Any]:
    return catalog_repo.load(catalog_id)


def save_plan(plan_id: str, plan: dict[str, Any]) -> Path:
    return plan_repo.save_raw(plan_id, plan)


def load_plan(plan_id: str) -> dict[str, Any]:
    return plan_repo.load(plan_id)


# ---- render jobs (no repo abstraction — simple enough as-is) --------------

def save_job(job_id: str, job: dict[str, Any]) -> Path:
    p = render_dir() / f"{job_id}.json"
    p.write_text(json.dumps(job, indent=2))
    return p


def load_job(job_id: str) -> dict[str, Any]:
    p = render_dir() / f"{job_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"job {job_id} not found")
    return json.loads(p.read_text())

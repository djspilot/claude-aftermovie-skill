"""Catalog + Plan repositories — the single Seam between the pipeline and
on-disk JSON state under `~/.skills-data/aftermovie/`.

Before these existed, three call sites (`pipeline_runner`, `select/server`,
`mcp_server`) reached past the shallow `state.py` helpers and reinvented the
same id-stamping (`_aftermovie.catalog_id` / `_aftermovie.plan_id`),
cache-hit branching, and atomic-copy bits each in their own way. The
repositories own those Implementation details so callers ask in domain
terms: *"do I have a catalog for this folder?"*, *"persist this plan."*

The on-disk layout and JSON shape are unchanged — `CatalogRepository` and
`PlanRepository` write the same files `state.save_catalog`/`save_plan` used
to write. The Interface is what's new.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from aftermovie import config


# ---- path helpers (kept private; callers go through the repos) -------------

def _catalogs_dir() -> Path:
    d = config.data_dir() / "catalogs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plans_dir() -> Path:
    d = config.data_dir() / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


# ---- CatalogRepository -----------------------------------------------------

class CatalogRepository:
    """Read/write Catalogs (the analyze-step output) keyed by `catalog_id`.

    A `catalog_id` is a content hash over the source folder path + the
    sorted file list + sizes + mtimes. Identical input therefore yields an
    identical id, which is what makes the cache lookup work — `get(folder)`
    returns the cached Catalog when the folder hasn't changed since the
    last analyze run.
    """

    def id_for(self, folder: Path) -> str:
        """Stable Catalog id derived from folder + sorted file list + mtimes."""
        folder = Path(folder).resolve()
        pieces = [str(folder)]
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                try:
                    st = f.stat()
                    pieces.append(f"{f.relative_to(folder)}|{st.st_size}|{int(st.st_mtime)}")
                except OSError:
                    continue
        return _hash("\n".join(pieces))

    def path_for_id(self, catalog_id: str) -> Path:
        """Where the Catalog with this id lives (or would live) on disk."""
        return _catalogs_dir() / f"{catalog_id}.json"

    def get(self, folder: Path) -> dict[str, Any] | None:
        """Cache lookup: return the stored Catalog for `folder`, or None.

        None means *miss* — caller should run analyze and then `put(...)`
        the result. Distinguished from `load(cid)`, which raises when the
        id is wrong (the MCP API surface still wants that exception).
        """
        cid = self.id_for(folder)
        p = self.path_for_id(cid)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def load(self, catalog_id: str) -> dict[str, Any]:
        """Direct fetch by id; raises FileNotFoundError if missing."""
        p = self.path_for_id(catalog_id)
        if not p.is_file():
            raise FileNotFoundError(f"catalog {catalog_id} not found")
        return json.loads(p.read_text())

    def put(self, folder: Path, catalog: dict[str, Any]) -> Path:
        """Stamp `_aftermovie.catalog_id` onto `catalog` and persist it.

        Returns the path the catalog landed at. The stamp lets downstream
        code (and the GUI's /api/plan endpoint) recover the id without
        re-walking the source tree.
        """
        cid = self.id_for(folder)
        catalog.setdefault("_aftermovie", {})["catalog_id"] = cid
        p = self.path_for_id(cid)
        p.write_text(json.dumps(catalog, indent=2))
        return p

    def copy_into(self, catalog_id: str, dest: Path) -> None:
        """Copy the cached catalog into a workdir for the score stage."""
        shutil.copy(self.path_for_id(catalog_id), dest)

    def save_raw(self, catalog_id: str, catalog: dict[str, Any]) -> Path:
        """Persist a Catalog by an already-known id (no stamping).

        The MCP `analyze_folder` tool computes the id up front (so it can
        return it to the client immediately) and then writes the catalog
        once the background job finishes — that path needs to write by id,
        not by folder. Everywhere else, prefer `put(folder, catalog)`.
        """
        p = self.path_for_id(catalog_id)
        p.write_text(json.dumps(catalog, indent=2))
        return p


# ---- PlanRepository --------------------------------------------------------

class PlanRepository:
    """Read/write Plans (the score-step output) keyed by `plan_id`.

    A `plan_id` is a content hash over (`catalog_id`, song path, theme,
    max_length, aspect, seed) — the inputs that uniquely determine a Plan.
    Unlike Catalogs, Plans aren't cache-skipped (scoring is cheap), but
    each Plan still gets stamped + saved so the GUI can locate it by
    catalog_id later.
    """

    def id_for(
        self,
        catalog_id: str,
        song_path: Path,
        theme: str | None,
        target_length_s: float | None,
        aspect: str,
        seed: int,
    ) -> str:
        parts = [
            catalog_id,
            str(Path(song_path).resolve()),
            theme or "",
            f"{target_length_s or ''}",
            aspect,
            str(seed),
        ]
        return _hash("|".join(parts))

    def path_for_id(self, plan_id: str) -> Path:
        return _plans_dir() / f"{plan_id}.json"

    def load(self, plan_id: str) -> dict[str, Any]:
        """Direct fetch by id; raises FileNotFoundError if missing."""
        p = self.path_for_id(plan_id)
        if not p.is_file():
            raise FileNotFoundError(f"plan {plan_id} not found")
        return json.loads(p.read_text())

    def put(
        self,
        catalog_id: str,
        song_path: Path,
        opts: "PlanIdOpts",
        plan: dict[str, Any],
    ) -> Path:
        """Stamp `_aftermovie.catalog_id` + `plan_id` onto `plan` and persist.

        `opts` carries the four scoring-input knobs that go into `plan_id`
        derivation (theme, max_length, aspect, seed). Returns the on-disk
        path so callers can echo it back to the workdir if they want.
        """
        pid = self.id_for(
            catalog_id, song_path, opts.theme, opts.max_length,
            opts.aspect, opts.seed,
        )
        tag = plan.setdefault("_aftermovie", {})
        tag["catalog_id"] = catalog_id
        tag["plan_id"] = pid
        p = self.path_for_id(pid)
        p.write_text(json.dumps(plan, indent=2))
        return p

    def save_raw(self, plan_id: str, plan: dict[str, Any]) -> Path:
        """Persist a Plan by an already-known id (no stamping).

        Used by `mcp_server.propose_plan` and `tweak_plan`, which compute
        the id themselves and embed it in the plan body's own `plan_id`
        field for backward compat with existing JSON files on disk.
        """
        p = self.path_for_id(plan_id)
        p.write_text(json.dumps(plan, indent=2))
        return p

    def get_latest_for_catalog(self, catalog_id: str) -> dict[str, Any] | None:
        """Return the most-recently-modified Plan tagged with this catalog_id.

        None when no matching Plan exists yet (i.e. the user hasn't kicked
        off a render from this folder). Backs the GUI's `/api/plan` endpoint.
        """
        d = _plans_dir()
        if not d.is_dir():
            return None
        candidates = sorted(
            d.glob("*.json"),
            key=lambda p: p.stat().st_mtime if p.is_file() else 0.0,
            reverse=True,
        )
        for p in candidates:
            try:
                plan = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            tag = plan.get("_aftermovie") if isinstance(plan, dict) else None
            if isinstance(tag, dict) and tag.get("catalog_id") == catalog_id:
                return plan
        return None


# ---- small value object for put(opts=...) ----------------------------------

class PlanIdOpts:
    """The four scoring-input knobs that participate in `plan_id` derivation.

    Pulled out as its own type so `PlanRepository.put` has a stable
    Interface — adding a new knob to the id (e.g. `pace`) only touches this
    class + `PlanRepository.id_for`, not every call site.
    """

    __slots__ = ("theme", "max_length", "aspect", "seed")

    def __init__(
        self,
        theme: str | None,
        max_length: float | None,
        aspect: str,
        seed: int = 0,
    ) -> None:
        self.theme = theme
        self.max_length = max_length
        self.aspect = aspect
        self.seed = seed


# ---- module-level singletons (stateless; safe to share) --------------------

catalog_repo = CatalogRepository()
plan_repo = PlanRepository()

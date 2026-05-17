"""FastMCP server exposing aftermovie's analyze → plan → render pipeline as tools.

The MCP server is the preferred interface for Claude Code; the CLI is the
fallback. Both call the same underlying functions in `aftermovie.analyze`,
`aftermovie.score`, and `aftermovie.render`.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from aftermovie.analyze.clip import analyze_clip, discover_sources
from aftermovie.config import (
    DEFAULT_TARGET_LEN_S,
    THEMES,
    VIDEO_EXTS,
    list_luts,
)
from aftermovie.effective_config import resolve as resolve_config
from aftermovie.env_config import load_env_file
from aftermovie.mcp_server import jobs
from aftermovie.pipeline_runner import AutoOpts, run_auto
from aftermovie.render.pipeline import cmd_render
from aftermovie.render.transitions import decide_transitions
from aftermovie.score.scorer import build_plan
from aftermovie.score.song import analyze_song
from aftermovie.state import (
    catalog_id_for,
    load_catalog,
    load_plan,
    plan_id_for,
    save_catalog,
    save_plan,
)

# Match the CLI: load the env file into the process env on import so render-
# time code paths that read os.environ.get(...) (e.g. AUDIO_INTEREST_THRESHOLD)
# see user-configured values. The actual config-resolution still happens
# through `resolve_config(...)` per tool call, so MCP and CLI cannot diverge.
load_env_file()

mcp = FastMCP("aftermovie")


# ---- read-only tools -------------------------------------------------------

@mcp.tool(description="List the built-in theme presets (LUT + audio + ramp policy).")
def list_themes() -> dict[str, Any]:
    return {"themes": [{"name": name, **{k: v for k, v in t.items()}}
                       for name, t in THEMES.items()]}


@mcp.tool(description="List the available color LUTs.")
def aftermovie_list_luts() -> dict[str, Any]:
    return {"luts": list_luts()}


@mcp.tool(description="Fetch a single clip's ClipInfo from a saved catalog.")
def inspect_clip(catalog_id: str, clip_index: int) -> dict[str, Any]:
    catalog = load_catalog(catalog_id)
    clips = catalog.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise IndexError(f"clip_index {clip_index} out of range (have {len(clips)})")
    return clips[clip_index]


@mcp.tool(description="Fetch a saved plan. If include_entries=False only return the summary.")
def get_plan(plan_id: str, include_entries: bool = True) -> dict[str, Any]:
    plan = load_plan(plan_id)
    if not include_entries:
        plan = {k: v for k, v in plan.items() if k != "entries"}
    return plan


# ---- jobs ------------------------------------------------------------------

@mcp.tool(description="Scan a folder for video clips + standalone stills "
                      "(HEIC/JPG/PNG materialized as 2.5s Ken-Burns clips). "
                      "Live Photo pairs (HEIC + same-stem MOV) keep only the MOV. "
                      "Returns a job_id; poll with get_job.")
def analyze_folder(
    path: str,
    force_reanalyze: bool = False,
    still_duration_s: float | None = None,
    include_stills: bool | None = None,
) -> dict[str, Any]:
    # Resolve still-handling knobs through the same chain the CLI uses.
    # None means "caller didn't pass a value" -> fall back to env/builtin.
    cfg = resolve_config(cli_overrides={
        "still_duration": still_duration_s,
        "no_stills": (not include_stills) if include_stills is not None else None,
    })
    still_duration_s = cfg.still_duration
    include_stills = not cfg.no_stills

    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    catalog_id = catalog_id_for(folder)

    if not force_reanalyze:
        try:
            load_catalog(catalog_id)
            return {"job_id": None, "catalog_id": catalog_id, "cached": True}
        except FileNotFoundError:
            pass

    files = discover_sources(folder, still_duration_s=still_duration_s,
                             include_stills=include_stills)
    if not files:
        raise FileNotFoundError(
            f"no usable files in {folder} "
            f"(videos: {sorted(VIDEO_EXTS)}; stills: heic/jpg/png — toggle with include_stills)"
        )

    def _work(cancel):
        clips: list[dict[str, Any]] = []
        warnings: list[str] = []
        for f in files:
            if cancel.is_set():
                break
            info = analyze_clip(f)
            if info is None:
                warnings.append(f"skipped {f.name}")
                continue
            clips.append(asdict(info))
        catalog = {"clips": clips, "folder": str(folder)}
        save_catalog(catalog_id, catalog)
        return {
            "catalog_id": catalog_id,
            "n_clips": len(clips),
            "n_skipped": len(warnings),
            "warnings": warnings,
        }

    job_id = jobs.start_job("analyze_folder", _work)
    return {"job_id": job_id, "catalog_id": catalog_id, "cached": False}


@mcp.tool(description="Propose an edit plan from a catalog + song + theme. "
                      "Synchronous — returns plan_id + summary.")
def propose_plan(
    catalog_id: str,
    song_path: str,
    theme: str | None = None,
    target_length_s: float | None = None,
    aspect: str | None = None,
    audio_mix: str | None = None,
    pace: str | None = None,
    transitions: str | None = None,
    titles: list[dict[str, Any]] | None = None,
    reframe: bool | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    # Resolve through the same precedence chain as the CLI: defaults -> env
    # file -> theme bundle -> caller-supplied kwargs. None kwargs fall
    # through to lower layers (so an MCP caller can lean on the env file).
    cfg = resolve_config(
        cli_overrides={
            "aspect": aspect,
            "audio_mix": audio_mix,
            "pace": pace,
            "transitions": transitions,
            "max_length": target_length_s,
            "no_reframe": (not reframe) if reframe is not None else None,
        },
        theme=theme,
    )

    catalog = load_catalog(catalog_id)
    song = analyze_song(Path(song_path).expanduser().resolve())
    requested = cfg.max_length or DEFAULT_TARGET_LEN_S
    target = min(song["duration_s"], requested)

    entries = build_plan(catalog, song, target_len=target,
                         no_speed_ramp=cfg.no_speed_ramp, pace=cfg.pace)
    if cfg.transitions in ("auto", "soft"):
        decide_transitions(entries, song, mode=cfg.transitions)

    plan_id = plan_id_for(catalog_id, Path(song_path), cfg.theme, target,
                          cfg.aspect, seed)
    plan = {
        "plan_id": plan_id,
        "catalog_id": catalog_id,
        "song": str(Path(song_path).expanduser().resolve()),
        "song_meta": song,
        "theme": cfg.theme or "cinematic",
        "target_length_s": target,
        "aspect": cfg.aspect,
        "resolution": cfg.res,
        "fps": cfg.fps,
        "lut": cfg.lut,
        "music_db": cfg.music_db,
        "clip_db": cfg.clip_db,
        "audio_mix": cfg.audio_mix,
        "pace": cfg.pace,
        "transitions": cfg.transitions,
        "titles": titles or [],
        "reframe": not cfg.no_reframe,
        "song_start_s": float(song["intro_end_s"]),
        "entries": entries,
    }
    save_plan(plan_id, plan)
    sources = sorted({e["source"] for e in entries})
    return {
        "plan_id": plan_id,
        "summary": {
            "n_cuts": len(entries),
            "total_length_s": target,
            "sources_used": len(sources),
            "bpm": round(song["tempo_bpm"], 1),
        },
    }


# ---- tweak ops -------------------------------------------------------------

@mcp.tool(description="Apply tweak operations to a plan; returns a new plan_id. "
                      "Ops: {op:'exclude_source', source}, {op:'set', path, value}, "
                      "{op:'swap', beat_index, with_candidate_rank}, "
                      "{op:'pin', beat_index, source, start_s, end_s}.")
def tweak_plan(plan_id: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    import copy

    plan = load_plan(plan_id)
    new_plan = copy.deepcopy(plan)
    diff: list[str] = []

    for op in ops:
        kind = op.get("op")
        if kind == "exclude_source":
            src = op["source"]
            before = len(new_plan["entries"])
            new_plan["entries"] = [e for e in new_plan["entries"] if e["source"] != src]
            diff.append(f"excluded {src} ({before - len(new_plan['entries'])} cuts removed)")
        elif kind == "set":
            path = op["path"]
            value = op["value"]
            _set_path(new_plan, path, value)
            diff.append(f"set {path}={value}")
        elif kind == "swap":
            idx = int(op["beat_index"])
            rank = int(op.get("with_candidate_rank", 1))
            _swap_at(new_plan, idx, rank)
            diff.append(f"swapped entry[{idx}] with rank {rank}")
        elif kind == "pin":
            idx = int(op["beat_index"])
            new_plan["entries"][idx]["source"] = op["source"]
            new_plan["entries"][idx]["start_s"] = float(op["start_s"])
            new_plan["entries"][idx]["end_s"] = float(op["end_s"])
            diff.append(f"pinned entry[{idx}] to {op['source']}")
        elif kind == "add_title":
            new_plan.setdefault("titles", []).append({
                "kind": op.get("kind", "intro"),
                "text": op.get("text", ""),
                "duration_s": float(op.get("duration_s", 2.0)),
                "at_s": float(op["at_s"]) if "at_s" in op else None,
            })
            diff.append(f"added title kind={op.get('kind')} text={op.get('text', '')!r}")
        else:
            raise ValueError(f"unknown op: {kind}")

    # new id: derive from old id + ops payload
    import hashlib
    import json as _json
    seed = plan_id + _json.dumps(ops, sort_keys=True)
    new_id = hashlib.sha1(seed.encode()).hexdigest()[:12]
    new_plan["plan_id"] = new_id
    save_plan(new_id, new_plan)
    return {"plan_id": new_id, "diff": diff}


def _set_path(obj: dict[str, Any], path: str, value: Any) -> None:
    """Tiny dotted/indexed path setter — 'music_db', 'entries[3].speed'."""
    parts: list[Any] = []
    for chunk in path.split("."):
        if "[" in chunk and chunk.endswith("]"):
            name, idx = chunk[:-1].split("[")
            parts.append(name)
            parts.append(int(idx))
        else:
            parts.append(chunk)
    cur: Any = obj
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def _swap_at(plan: dict[str, Any], idx: int, rank: int) -> None:
    """Replace entries[idx] with the rank-th highest-scoring entry from later in the plan."""
    entries = plan["entries"]
    if idx >= len(entries):
        raise IndexError(f"entry index {idx} out of range")
    pool = sorted(
        ((i, e) for i, e in enumerate(entries) if i != idx),
        key=lambda kv: kv[1].get("score", 0),
        reverse=True,
    )
    if not pool or rank > len(pool):
        raise ValueError(f"no entry of rank {rank} available to swap")
    j, other = pool[rank - 1]
    entries[idx], entries[j] = entries[j], entries[idx]


# ---- render job -----------------------------------------------------------

@mcp.tool(description="Render a plan to an MP4. Returns a job_id; poll with get_job.")
def render_plan(plan_id: str, output_path: str) -> dict[str, Any]:
    plan = load_plan(plan_id)
    out = Path(output_path).expanduser().resolve()

    def _work(cancel):
        # Write plan to a temp file for cmd_render's argparse contract.
        import json as _json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write(_json.dumps(plan))
            plan_tmp = tf.name
        try:
            args = argparse.Namespace(plan=plan_tmp, output=str(out))
            cmd_render(args)
        finally:
            try:
                Path(plan_tmp).unlink()
            except OSError:
                pass

        from aftermovie.ffmpeg_cmd import ffprobe_json
        info = ffprobe_json(out)
        return {
            "output_path": str(out),
            "duration_s": float(info.get("format", {}).get("duration", 0)),
            "streams": [
                {"codec_type": s.get("codec_type"), "codec_name": s.get("codec_name")}
                for s in info.get("streams", [])
            ],
        }

    job_id = jobs.start_job("render_plan", _work)
    return {"job_id": job_id}


# ---- one-shot auto --------------------------------------------------------

@mcp.tool(description="One-shot analyze → score → render for a clips folder + song. "
                      "Same orchestration as the CLI's `aftermovie auto`. "
                      "Returns a job_id; poll with get_job. Theme bundles override "
                      "only the knobs the caller left at their built-in defaults.")
def auto(
    clips: str,
    song: str,
    output: str,
    theme: str | None = None,
    aspect: str = "16:9",
    res: str = "1920x1080",
    fps: int = 30,
    max_length: float | None = None,
    still_duration: float = 2.5,
    no_stills: bool = False,
    audio_mix: str = "ducked",
    music_db: float = -8.0,
    clip_db: float = -18.0,
    pace: str = "medium",
    transitions: str = "cut",
    no_speed_ramp: bool = False,
    no_reframe: bool = False,
    lut: str | None = None,
    titles: str | None = None,
    title_text: str | None = None,
) -> dict[str, Any]:
    clips_path = Path(clips).expanduser().resolve()
    song_path = Path(song).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not clips_path.is_dir():
        raise FileNotFoundError(f"not a directory: {clips_path}")
    if not song_path.is_file():
        raise FileNotFoundError(f"song not found: {song_path}")

    opts = AutoOpts(
        aspect=aspect,
        res=res,
        fps=fps,
        max_length=max_length,
        still_duration=still_duration,
        no_stills=no_stills,
        audio_mix=audio_mix,
        music_db=music_db,
        clip_db=clip_db,
        pace=pace,
        transitions=transitions,
        no_speed_ramp=no_speed_ramp,
        no_reframe=no_reframe,
        lut=lut,
        theme=theme,
        titles=titles,
        title_text=title_text,
    )

    def _work(cancel):
        run_auto(clips_path, song_path, output_path, opts)
        from aftermovie.ffmpeg_cmd import ffprobe_json
        info = ffprobe_json(output_path)
        return {
            "output_path": str(output_path),
            "duration_s": float(info.get("format", {}).get("duration", 0)),
            "streams": [
                {"codec_type": s.get("codec_type"), "codec_name": s.get("codec_name")}
                for s in info.get("streams", [])
            ],
        }

    job_id = jobs.start_job("auto", _work)
    return {"job_id": job_id}


@mcp.tool(description="Get the status (and result, when finished) of a job.")
def get_job(job_id: str) -> dict[str, Any]:
    return jobs.get_status(job_id)


@mcp.tool(description="Request cancellation of a running job. Returns whether the cancel flag was set.")
def cancel_job(job_id: str) -> dict[str, Any]:
    return {"cancelled": jobs.cancel(job_id)}


# ---- entrypoint -----------------------------------------------------------

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

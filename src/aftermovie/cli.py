"""Argparse CLI — dispatcher for analyze/score/render/auto."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from aftermovie.analyze.clip import cmd_analyze
from aftermovie.config import (
    DEFAULT_CLIP_DB,
    DEFAULT_FPS,
    DEFAULT_MUSIC_DB,
    DEFAULT_RES,
    THEMES,
)
from aftermovie.ffmpeg_cmd import log
from aftermovie.render.pipeline import cmd_render
from aftermovie.score.scorer import cmd_score


def cmd_auto(args: argparse.Namespace) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="aftermovie_auto_"))
    log(f"Working dir: {workdir}")
    catalog_path = workdir / "catalog.json"
    plan_path = workdir / "plan.json"

    a = argparse.Namespace(
        clips=args.clips,
        out=str(catalog_path),
        still_duration=getattr(args, "still_duration", 2.5),
        no_stills=getattr(args, "no_stills", False),
    )
    cmd_analyze(a)

    s = argparse.Namespace(
        catalog=str(catalog_path),
        song=args.song,
        out=str(plan_path),
        max_length=args.max_length,
        aspect=args.aspect,
        res=args.res,
        fps=args.fps,
        lut=args.lut,
        music_db=args.music_db,
        clip_db=args.clip_db,
        no_speed_ramp=args.no_speed_ramp,
        audio_mix=getattr(args, "audio_mix", "music_only"),
        transitions=getattr(args, "transitions", "cut"),
        titles=getattr(args, "titles", None),
        title_text=getattr(args, "title_text", None),
        no_reframe=getattr(args, "no_reframe", False),
    )
    cmd_score(s)

    r = argparse.Namespace(plan=str(plan_path), output=args.output)
    cmd_render(r)

    log(f"Intermediate files preserved in: {workdir}")


def _add_score_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-length", type=float, default=None)
    p.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "1:1"])
    p.add_argument("--res", default=DEFAULT_RES)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--lut", default=None)
    p.add_argument("--music-db", type=float, default=DEFAULT_MUSIC_DB)
    p.add_argument("--clip-db", type=float, default=DEFAULT_CLIP_DB)
    p.add_argument("--no-speed-ramp", action="store_true")
    p.add_argument("--audio-mix", default="music_only",
                   choices=["music_only", "ducked", "clip_only"],
                   help="How to mix audio: music only, music+clip ducked, or clip only.")
    p.add_argument("--transitions", default="cut", choices=["cut", "auto"],
                   help="cut = hard cuts only; auto = let the scorer pick crossfade/whip.")
    p.add_argument("--titles", default=None,
                   help="Comma-separated list of title kinds (intro,outro).")
    p.add_argument("--title-text", default=None,
                   help="Title text applied to intro/outro cards.")
    p.add_argument("--no-reframe", action="store_true",
                   help="Disable face-aware reframing on 9:16 output.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aftermovie",
        description="Beat-synced highlight video generator.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="Scan a folder of clips and extract features.")
    pa.add_argument("--clips", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--still-duration", type=float, default=2.5,
                    help="Per-still clip duration (s) for HEIC/JPG/PNG materials.")
    pa.add_argument("--no-stills", action="store_true",
                    help="Ignore HEIC/JPG/PNG stills; analyze only native video.")
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("score", help="Build an edit plan from a catalog and song.")
    ps.add_argument("--catalog", required=True)
    ps.add_argument("--song", required=True)
    ps.add_argument("--out", required=True)
    _add_score_flags(ps)
    ps.set_defaults(func=cmd_score)

    pr = sub.add_parser("render", help="Execute a plan via ffmpeg.")
    pr.add_argument("--plan", required=True)
    pr.add_argument("--output", required=True)
    pr.set_defaults(func=cmd_render)

    pu = sub.add_parser("auto", help="One-shot: analyze → score → render.")
    pu.add_argument("--clips", required=True)
    pu.add_argument("--song", required=True)
    pu.add_argument("--output", required=True)
    pu.add_argument("--still-duration", type=float, default=2.5,
                    help="Per-still clip duration (s) for HEIC/JPG/PNG materials.")
    pu.add_argument("--no-stills", action="store_true",
                    help="Ignore HEIC/JPG/PNG stills; analyze only native video.")
    _add_score_flags(pu)
    pu.add_argument("--theme", default=None,
                    choices=sorted(THEMES.keys()),
                    help="Preset bundle (cinematic, punchy, chill, nostalgic).")
    pu.set_defaults(func=cmd_auto)

    pd = sub.add_parser("doctor", help="Check environment (ffmpeg, deps, LUTs).")
    pd.set_defaults(func=cmd_doctor)

    return p


def cmd_doctor(args: argparse.Namespace) -> None:
    """Self-check for ffmpeg, python deps, LUTs, optional MCP/mediapipe."""
    import shutil

    from aftermovie.config import lut_dir

    print("aftermovie doctor")
    print("-----------------")
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    print(f"  ffmpeg:        {'OK ' + ff if ff else 'MISSING'}")
    print(f"  ffprobe:       {'OK ' + fp if fp else 'MISSING'}")

    for mod in ("librosa", "numpy", "soundfile", "scipy"):
        try:
            __import__(mod)
            print(f"  {mod:14s} OK")
        except ImportError as e:
            print(f"  {mod:14s} MISSING ({e})")

    for opt, label in [("mcp", "mcp (optional)"), ("mediapipe", "mediapipe (optional)")]:
        try:
            __import__(opt)
            print(f"  {label:24s} OK")
        except ImportError:
            print(f"  {label:24s} not installed")

    from aftermovie.analyze.faces import available as faces_available
    print(f"  faces feature:           {'OK' if faces_available() else 'unavailable (mediapipe or model missing)'}")

    lut_d = lut_dir()
    luts = list(lut_d.glob("*.cube")) if lut_d.is_dir() else []
    print(f"  LUTs:          {len(luts)} found in {lut_d}")
    for l in luts:
        print(f"    - {l.name}")


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "theme", None):
        preset = THEMES.get(args.theme, {})
        for k, v in preset.items():
            if k == "description":
                continue
            cur = getattr(args, k, None)
            # Apply preset only if the user didn't explicitly set the flag.
            # `lut` defaults to None; `music_db` defaults to DEFAULT_MUSIC_DB.
            if k == "lut" and cur is None:
                setattr(args, k, v)
            elif k == "music_db" and cur == DEFAULT_MUSIC_DB:
                setattr(args, k, v)
            elif k == "no_speed_ramp" and cur is False:
                setattr(args, k, v)
    args.func(args)

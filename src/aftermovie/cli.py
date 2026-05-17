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
from aftermovie.env_config import (
    DEFAULT_CONFIG_TEMPLATE,
    config_path,
    env_bool,
    env_float,
    env_int,
    env_str,
    load_env_file,
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
        audio_mix=getattr(args, "audio_mix", "ducked"),
        pace=getattr(args, "pace", "medium"),
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
    p.add_argument("--max-length", "--length", dest="max_length", type=float,
                   default=env_float("AFTERMOVIE_MAX_LENGTH", None),
                   help="Target output length in seconds (default: min(song, 90)). "
                        "Env: AFTERMOVIE_MAX_LENGTH.")
    p.add_argument("--aspect", default=env_str("AFTERMOVIE_ASPECT", "16:9"),
                   choices=["16:9", "9:16", "1:1"])
    p.add_argument("--res", default=env_str("AFTERMOVIE_RES", DEFAULT_RES))
    p.add_argument("--fps", type=int, default=env_int("AFTERMOVIE_FPS", DEFAULT_FPS))
    p.add_argument("--lut", default=env_str("AFTERMOVIE_LUT", None))
    p.add_argument("--music-db", type=float,
                   default=env_float("AFTERMOVIE_MUSIC_DB", DEFAULT_MUSIC_DB))
    p.add_argument("--clip-db", type=float,
                   default=env_float("AFTERMOVIE_CLIP_DB", DEFAULT_CLIP_DB))
    p.add_argument("--no-speed-ramp", action="store_true",
                   default=env_bool("AFTERMOVIE_NO_SPEED_RAMP", False))
    p.add_argument("--audio-mix", default=env_str("AFTERMOVIE_AUDIO_MIX", "ducked"),
                   choices=["music_only", "ducked", "clip_only"],
                   help="How to mix audio: ducked (default — music + clip with "
                        "voice-band sidechain), music_only, or clip_only. "
                        "Env: AFTERMOVIE_AUDIO_MIX.")
    p.add_argument("--pace", default=env_str("AFTERMOVIE_PACE", "medium"),
                   choices=["fast", "medium", "slow", "auto"],
                   help="fast = every beat (~0.5s cuts at 100bpm), "
                        "medium (default) = every 2nd beat, "
                        "slow = every 4th beat (downbeats only), "
                        "auto = energy-aware (Quik-style: tight on drops, breathes on verses). "
                        "Env: AFTERMOVIE_PACE.")
    p.add_argument("--transitions", default=env_str("AFTERMOVIE_TRANSITIONS", "cut"),
                   choices=["cut", "auto", "soft"],
                   help="cut = hard cuts only; auto = scorer-picked crossfade/whip; "
                        "soft = short crossfade on every cut. "
                        "Env: AFTERMOVIE_TRANSITIONS.")
    p.add_argument("--titles", default=None,
                   help="Comma-separated list of title kinds (intro,outro).")
    p.add_argument("--title-text", default=None,
                   help="Title text applied to intro/outro cards.")
    p.add_argument("--no-reframe", action="store_true",
                   default=env_bool("AFTERMOVIE_NO_REFRAME", False),
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
    pa.add_argument("--still-duration", type=float,
                    default=env_float("AFTERMOVIE_STILL_DURATION", 2.5),
                    help="Per-still clip duration (s) for HEIC/JPG/PNG materials.")
    pa.add_argument("--no-stills", action="store_true",
                    default=env_bool("AFTERMOVIE_NO_STILLS", False),
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
    pu.add_argument("--theme", default=env_str("AFTERMOVIE_THEME", None),
                    choices=sorted(THEMES.keys()),
                    help="Preset bundle (cinematic, punchy, chill, nostalgic). "
                         "Env: AFTERMOVIE_THEME.")
    pu.set_defaults(func=cmd_auto)

    pd = sub.add_parser("doctor", help="Check environment (ffmpeg, deps, LUTs).")
    pd.set_defaults(func=cmd_doctor)

    pc = sub.add_parser("init-config",
                        help="Write a default env file at ~/.aftermovie/aftermovie.env.")
    pc.add_argument("--force", action="store_true",
                    help="Overwrite an existing config file.")
    pc.set_defaults(func=cmd_init_config)

    pl = sub.add_parser("show-config",
                        help="Print the effective defaults (from env file + builtins).")
    pl.set_defaults(func=cmd_show_config)

    return p


def cmd_init_config(args: argparse.Namespace) -> None:
    target = config_path()
    if target.exists() and not args.force:
        print(f"Config already exists at {target} (pass --force to overwrite).")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_CONFIG_TEMPLATE)
    print(f"Wrote default config to {target}")
    print("Edit it to change defaults for `aftermovie auto`. CLI flags still win.")


def cmd_show_config(args: argparse.Namespace) -> None:
    p = config_path()
    print(f"Config file: {p} ({'present' if p.is_file() else 'missing — run `aftermovie init-config`'})")
    print()
    print("Effective defaults (env file + builtins, before CLI flags):")
    rows = [
        ("AFTERMOVIE_THEME",          env_str("AFTERMOVIE_THEME", "(none)")),
        ("AFTERMOVIE_ASPECT",         env_str("AFTERMOVIE_ASPECT", "16:9")),
        ("AFTERMOVIE_RES",            env_str("AFTERMOVIE_RES", DEFAULT_RES)),
        ("AFTERMOVIE_FPS",            env_int("AFTERMOVIE_FPS", DEFAULT_FPS)),
        ("AFTERMOVIE_MAX_LENGTH",     env_float("AFTERMOVIE_MAX_LENGTH", None)),
        ("AFTERMOVIE_STILL_DURATION", env_float("AFTERMOVIE_STILL_DURATION", 2.5)),
        ("AFTERMOVIE_AUDIO_MIX",      env_str("AFTERMOVIE_AUDIO_MIX", "ducked")),
        ("AFTERMOVIE_MUSIC_DB",       env_float("AFTERMOVIE_MUSIC_DB", DEFAULT_MUSIC_DB)),
        ("AFTERMOVIE_CLIP_DB",        env_float("AFTERMOVIE_CLIP_DB", DEFAULT_CLIP_DB)),
        ("AFTERMOVIE_PACE",           env_str("AFTERMOVIE_PACE", "medium")),
        ("AFTERMOVIE_TRANSITIONS",    env_str("AFTERMOVIE_TRANSITIONS", "cut")),
        ("AFTERMOVIE_NO_SPEED_RAMP",  env_bool("AFTERMOVIE_NO_SPEED_RAMP", False)),
        ("AFTERMOVIE_NO_STILLS",      env_bool("AFTERMOVIE_NO_STILLS", False)),
        ("AFTERMOVIE_NO_REFRAME",     env_bool("AFTERMOVIE_NO_REFRAME", False)),
    ]
    for k, v in rows:
        print(f"  {k:30s} {v}")
    theme = env_str("AFTERMOVIE_THEME", None)
    if theme and theme in THEMES:
        print()
        print(f"Theme bundle for '{theme}' (each value only applied if user didn't override):")
        for k, v in THEMES[theme].items():
            if k == "description":
                continue
            print(f"  {k:30s} {v}")
        print(f"  -- {THEMES[theme].get('description', '')}")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Self-check for ffmpeg, python deps, LUTs, optional MCP/mediapipe."""
    import shutil

    from aftermovie.config import lut_dir

    print("aftermovie doctor")
    print("-----------------")
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    et = shutil.which("exiftool")
    print(f"  ffmpeg:        {'OK ' + ff if ff else 'MISSING'}")
    print(f"  ffprobe:       {'OK ' + fp if fp else 'MISSING'}")
    print(f"  exiftool:      {'OK ' + et if et else 'not installed (single-file Live Photos will be used as stills only)'}")

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
    # Load user config first so env-backed argparse defaults pick it up.
    load_env_file()
    args = build_parser().parse_args()
    if getattr(args, "theme", None):
        _apply_theme(args)
    args.func(args)


# Per-flag "is this the built-in default?" check — used to decide whether the
# theme's value should override. Anything the user explicitly set on the CLI
# (or via the env file) stays untouched.
_THEME_DEFAULTS = {
    "lut": None,
    "music_db": DEFAULT_MUSIC_DB,
    "no_speed_ramp": False,
    "transitions": "cut",
    "audio_mix": "ducked",
    "pace": "medium",
}


def _apply_theme(args: argparse.Namespace) -> None:
    preset = THEMES.get(args.theme, {})
    for k, v in preset.items():
        if k == "description":
            continue
        cur = getattr(args, k, None)
        baseline = _THEME_DEFAULTS.get(k)
        # Only override when the field is still at the built-in default — that
        # means the user did not set it on the CLI or in the env file.
        if cur == baseline:
            setattr(args, k, v)

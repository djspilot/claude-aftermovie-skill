"""Argparse CLI — dispatcher for analyze/score/render/auto.

The CLI is intentionally thin: argparse gathers user-supplied values, then
`effective_config.resolve(...)` composes them with the env file, the chosen
theme bundle, and built-in defaults. Both this module and the MCP server use
the same `EffectiveConfig`, so they can't drift.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from aftermovie.analyze.clip import cmd_analyze
from aftermovie.config import THEMES
from aftermovie.effective_config import EffectiveConfig, resolve
from aftermovie.ffmpeg_cmd import log
from aftermovie.env_config import (
    DEFAULT_CONFIG_TEMPLATE,
    config_path,
    load_env_file,
)
from aftermovie.pipeline_runner import (
    opts_from_namespace,
    run_auto,
    run_render_only,
)
from aftermovie.render.pipeline import cmd_render
from aftermovie.score.scorer import cmd_score


def _cli_overrides_from(args: argparse.Namespace) -> dict[str, object]:
    """Pluck argparse-set fields into a `cli_overrides` dict for `resolve()`.

    Argparse defaults are None across the board (see `_add_score_flags`),
    so anything non-None here was set explicitly on the command line. Boolean
    flags use `store_const(True)` with default=None so an unset flag stays
    None (i.e. 'fall through to lower layer'), while `--no-speed-ramp` /
    `--no-reframe` on the command line shows up as True.
    """
    keys = (
        "aspect", "res", "fps", "max_length", "still_duration", "no_stills",
        "audio_mix", "music_db", "clip_db", "pace", "transitions",
        "no_speed_ramp", "no_reframe", "lut",
        "source_cap", "chronological", "preview",
    )
    out: dict[str, object] = {}
    for k in keys:
        if hasattr(args, k):
            v = getattr(args, k)
            if v is not None:
                out[k] = v
    return out


def _resolved_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Apply `EffectiveConfig.resolve` and project the result back onto a
    Namespace, preserving subparser-specific fields (clips/song/output/etc.).
    """
    theme = getattr(args, "theme", None)
    overrides = _cli_overrides_from(args)
    cfg = resolve(cli_overrides=overrides, theme=theme)

    # Copy through the resolved fields onto args so downstream cmd_* funcs,
    # which read from args, see the fully-composed values. We don't mutate
    # `clips`, `song`, `output`, etc.
    resolved = asdict(cfg)
    for k, v in resolved.items():
        setattr(args, k, v)
    # `theme` is also part of EffectiveConfig — preserve it on args too.
    setattr(args, "theme", cfg.theme)
    return args


def cmd_auto(args: argparse.Namespace) -> None:
    # Resolve env file + theme + CLI flags into a single config first,
    # then delegate to the shared pipeline_runner so the CLI and MCP
    # surfaces drive the same code path.
    args = _resolved_namespace(args)
    from_plan = getattr(args, "from_plan", None)
    if from_plan:
        # Skip analyze + score — the plan already encodes clips, song, and
        # all scoring decisions. Only the output path + reveal honour the
        # CLI namespace; everything else is read from plan.json by cmd_render.
        plan_path = Path(from_plan).expanduser().resolve()
        if not plan_path.is_file():
            raise SystemExit(f"--from-plan path is not a file: {plan_path}")
        if not getattr(args, "output", None):
            out_dir = Path(getattr(args, "output_dir", "")
                           or str(Path.home() / "Downloads")).expanduser()
            args.output = str(out_dir / f"aftermovie-{plan_path.stem}.mp4")
            log(f"Output → {args.output}")
        opts = opts_from_namespace(args)
        if getattr(args, "no_reveal", None):
            opts.reveal = False
        run_render_only(plan_path, Path(args.output), opts)
        return
    if not getattr(args, "clips", None) or not getattr(args, "song", None):
        raise SystemExit(
            "auto requires --clips and --song (or pass --from-plan PATH)."
        )
    if not getattr(args, "output", None):
        out_dir = Path(getattr(args, "output_dir", "")
                       or str(Path.home() / "Downloads")).expanduser()
        clips_name = Path(args.clips).expanduser().resolve().name or "edit"
        args.output = str(out_dir / f"aftermovie-{clips_name}.mp4")
        log(f"Output → {args.output}")
    opts = opts_from_namespace(args)
    if getattr(args, "no_reveal", None):
        opts.reveal = False
    run_auto(Path(args.clips), Path(args.song), Path(args.output), opts)


def _cmd_score_resolved(args: argparse.Namespace) -> None:
    """Wraps `cmd_score` so direct `score` invocations also get resolution."""
    args = _resolved_namespace(args)
    cmd_score(args)


def _cmd_analyze_resolved(args: argparse.Namespace) -> None:
    """Resolve still_duration / no_stills before delegating to cmd_analyze."""
    cfg = resolve(cli_overrides={
        "still_duration": getattr(args, "still_duration", None),
        "no_stills": getattr(args, "no_stills", None),
    })
    args.still_duration = cfg.still_duration
    args.no_stills = cfg.no_stills
    cmd_analyze(args)


def _add_score_flags(p: argparse.ArgumentParser) -> None:
    # All argparse defaults are None so they don't masquerade as user input
    # in `_cli_overrides_from`. The real defaults live in
    # `effective_config.BUILTIN_DEFAULTS` and flow through `resolve()`.
    p.add_argument("--max-length", "--length", dest="max_length", type=float,
                   default=None,
                   help="Target output length in seconds (default: min(song, 90)). "
                        "Env: AFTERMOVIE_MAX_LENGTH.")
    p.add_argument("--aspect", default=None,
                   choices=["16:9", "9:16", "1:1"])
    p.add_argument("--res", default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--lut", default=None)
    p.add_argument("--music-db", type=float, default=None)
    p.add_argument("--clip-db", type=float, default=None)
    p.add_argument("--no-speed-ramp", dest="no_speed_ramp",
                   action="store_const", const=True, default=None)
    p.add_argument("--audio-mix", default=None,
                   choices=["music_only", "ducked", "clip_only"],
                   help="How to mix audio: ducked (default — music + clip with "
                        "voice-band sidechain), music_only, or clip_only. "
                        "Env: AFTERMOVIE_AUDIO_MIX.")
    p.add_argument("--pace", default=None,
                   choices=["fast", "medium", "slow", "auto"],
                   help="fast = every beat (~0.5s cuts at 100bpm), "
                        "medium (default) = every 2nd beat, "
                        "slow = every 4th beat (downbeats only), "
                        "auto = energy-aware (Quik-style: tight on drops, breathes on verses). "
                        "Env: AFTERMOVIE_PACE.")
    p.add_argument("--transitions", default=None,
                   choices=["cut", "auto", "soft"],
                   help="cut = hard cuts only; auto = scorer-picked crossfade/whip; "
                        "soft = short crossfade on every cut. "
                        "Env: AFTERMOVIE_TRANSITIONS.")
    p.add_argument("--titles", default=None,
                   help="Comma-separated list of title kinds (intro,outro).")
    p.add_argument("--title-text", default=None,
                   help="Title text applied to intro/outro cards.")
    p.add_argument("--no-reframe", dest="no_reframe",
                   action="store_const", const=True, default=None,
                   help="Disable face-aware reframing on 9:16 output.")
    p.add_argument("--source-cap", dest="source_cap", type=int, default=None,
                   help="Max times a source file may appear in the plan. "
                        "1 = no duplicates (default). Env: AFTERMOVIE_SOURCE_CAP.")
    p.add_argument("--no-chronological", dest="chronological",
                   action="store_const", const=False, default=None,
                   help="Don't re-order picks by EXIF/creation time. "
                        "Env: AFTERMOVIE_CHRONOLOGICAL.")
    p.add_argument("--burst-window-s", dest="burst_window_s", type=float,
                   default=None,
                   help="Seconds used to collapse near-duplicate burst shots. "
                        "0 disables burst suppression.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aftermovie",
        description="Beat-synced highlight video generator.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="Scan a folder of clips and extract features.")
    pa.add_argument("--clips", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--still-duration", type=float, default=None,
                    help="Per-still clip duration (s) for HEIC/JPG/PNG materials.")
    pa.add_argument("--no-stills", dest="no_stills",
                    action="store_const", const=True, default=None,
                    help="Ignore HEIC/JPG/PNG stills; analyze only native video.")
    pa.set_defaults(func=_cmd_analyze_resolved)

    ps = sub.add_parser("score", help="Build an edit plan from a catalog and song.")
    ps.add_argument("--catalog", required=True)
    ps.add_argument("--song", required=True)
    ps.add_argument("--out", required=True)
    _add_score_flags(ps)
    ps.set_defaults(func=_cmd_score_resolved)

    pr = sub.add_parser("render", help="Execute a plan via ffmpeg.")
    pr.add_argument("--plan", required=True)
    pr.add_argument("--output", required=True)
    pr.set_defaults(func=cmd_render)

    # Alias for discoverability — `render-from-plan` reads more clearly in
    # docs and chat transcripts than the bare `render` verb. Delegates to
    # the same handler so behaviour is identical.
    prfp = sub.add_parser(
        "render-from-plan",
        help="Alias for `render` — execute an existing plan.json.",
    )
    prfp.add_argument("--plan", required=True)
    prfp.add_argument("--output", required=True)
    prfp.set_defaults(func=cmd_render)

    pu = sub.add_parser("auto", help="One-shot: analyze → score → render.")
    pu.add_argument("--clips", default=None,
                    help="Folder of source clips. Required unless --from-plan "
                         "is given (in which case the plan already encodes it).")
    pu.add_argument("--song", default=None,
                    help="Song path. Required unless --from-plan is given.")
    pu.add_argument("--from-plan", dest="from_plan", default=None,
                    help="Skip analyze + score; render the supplied plan.json "
                         "directly. --clips/--song become optional.")
    pu.add_argument("--output", default=None,
                    help="Output path. Defaults to "
                         "<AFTERMOVIE_OUTPUT_DIR>/aftermovie-<source>.mp4 "
                         "(typically ~/Downloads/).")
    pu.add_argument("--still-duration", type=float, default=None,
                    help="Per-still clip duration (s) for HEIC/JPG/PNG materials.")
    pu.add_argument("--no-stills", dest="no_stills",
                    action="store_const", const=True, default=None,
                    help="Ignore HEIC/JPG/PNG stills; analyze only native video.")
    _add_score_flags(pu)
    pu.add_argument("--theme", default=None,
                    choices=sorted(THEMES.keys()),
                    help="Preset bundle (cinematic, punchy, chill, nostalgic). "
                         "Env: AFTERMOVIE_THEME.")
    pu.add_argument("--preview", dest="preview",
                    action="store_const", const=True, default=None,
                    help="Fast-iteration render: quarter-res, 24fps, no LUT, "
                         "no reframe. ~5-8s per render. "
                         "Env: AFTERMOVIE_PREVIEW.")
    pu.add_argument("--no-reveal", dest="no_reveal",
                    action="store_const", const=True, default=None,
                    help="Skip the macOS notification + Finder reveal at end.")
    pu.add_argument("--force-analyze", dest="force_reanalyze",
                    action="store_true", default=False,
                    help="Bypass the on-disk catalog cache and re-run analyze "
                         "even if a catalog already exists for this clips folder.")
    pu.set_defaults(func=cmd_auto)

    psl = sub.add_parser("select",
                         help="Open a browser UI to pick clips before rendering.")
    psl.add_argument("--clips", required=True,
                     help="Folder of source clips. The chosen selection is "
                          "saved as <clips>/.aftermovie-selection.json.")
    psl.add_argument("--song", default=None,
                     help="Optional song path used when the GUI starts a render.")
    psl.add_argument("--port", type=int, default=8765,
                     help="HTTP port to bind locally (default: 8765).")
    psl.add_argument("--no-open", dest="no_open", action="store_true",
                     help="Don't auto-open the browser to the server URL.")
    psl.set_defaults(func=cmd_select)

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


def cmd_select(args: argparse.Namespace) -> None:
    """Boot the `aftermovie select` web GUI and block until Ctrl-C.

    The server is local-only (127.0.0.1) so the user's mixed-media folder
    never leaves the machine. Browsers auto-open via `open <url>` on macOS
    unless `--no-open` is passed (useful for headless testing).
    """
    from aftermovie.select.server import run as run_server

    clips = Path(args.clips).expanduser().resolve()
    if not clips.is_dir():
        raise SystemExit(f"--clips path is not a directory: {clips}")
    song = Path(args.song).expanduser().resolve() if args.song else None
    run_server(clips, port=args.port, song=song,
               open_browser=not args.no_open)


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
    # Resolve with no overrides + no explicit theme to see what env+builtins compose to.
    cfg_env_only = resolve(cli_overrides=None, theme=None)
    print("Effective defaults (env file + builtins, before CLI flags or theme):")
    rows: list[tuple[str, object]] = [
        ("AFTERMOVIE_THEME",          cfg_env_only.theme or "(none)"),
        ("AFTERMOVIE_ASPECT",         cfg_env_only.aspect),
        ("AFTERMOVIE_RES",            cfg_env_only.res),
        ("AFTERMOVIE_FPS",            cfg_env_only.fps),
        ("AFTERMOVIE_MAX_LENGTH",     cfg_env_only.max_length),
        ("AFTERMOVIE_STILL_DURATION", cfg_env_only.still_duration),
        ("AFTERMOVIE_AUDIO_MIX",      cfg_env_only.audio_mix),
        ("AFTERMOVIE_MUSIC_DB",       cfg_env_only.music_db),
        ("AFTERMOVIE_CLIP_DB",        cfg_env_only.clip_db),
        ("AFTERMOVIE_PACE",           cfg_env_only.pace),
        ("AFTERMOVIE_TRANSITIONS",    cfg_env_only.transitions),
        ("AFTERMOVIE_NO_SPEED_RAMP",  cfg_env_only.no_speed_ramp),
        ("AFTERMOVIE_NO_STILLS",      cfg_env_only.no_stills),
        ("AFTERMOVIE_NO_REFRAME",     cfg_env_only.no_reframe),
    ]
    for k, v in rows:
        print(f"  {k:30s} {v}")
    theme = cfg_env_only.theme
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
    # Load user config first so env-backed defaults (and any code that still
    # reads os.environ at render time) pick it up. EffectiveConfig also reads
    # the file directly, but this keeps backwards-compat for anything that
    # calls os.environ.get(...) downstream (e.g. render.pipeline).
    load_env_file()
    args = build_parser().parse_args()
    args.func(args)


__all__ = [
    "EffectiveConfig",
    "build_parser",
    "cmd_auto",
    "cmd_init_config",
    "cmd_show_config",
    "main",
]

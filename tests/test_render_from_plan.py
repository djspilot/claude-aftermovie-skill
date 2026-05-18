"""Tests for the `--from-plan` short-circuit and `render-from-plan` alias.

When a saved plan.json already exists, `aftermovie auto --from-plan <p>` (and
the dedicated `render-from-plan` subcommand) must skip analyze + score and
hand the plan straight to `cmd_render`. These tests stub the per-stage entry
points to assert the analyze/score functions are never called and the plan
path threads through to the render stage unmodified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aftermovie import cli, pipeline_runner
from aftermovie.pipeline_runner import AutoOpts, run_render_only


def _patch_pipeline(monkeypatch) -> dict[str, argparse.Namespace]:
    """Replace cmd_analyze/score/render with recorders.

    Mirrors the helper in test_pipeline_runner.py so assertions stay
    consistent: any key that lands in the returned dict means that stage
    actually executed.
    """
    captured: dict[str, argparse.Namespace] = {}

    def fake_analyze(args: argparse.Namespace) -> None:
        captured["analyze"] = args
        Path(args.out).write_text(json.dumps({"clips": []}))

    def fake_score(args: argparse.Namespace) -> None:
        captured["score"] = args
        Path(args.out).write_text(json.dumps({"entries": []}))

    def fake_render(args: argparse.Namespace) -> None:
        captured["render"] = args
        Path(args.output).write_bytes(b"")

    monkeypatch.setattr(pipeline_runner, "cmd_analyze", fake_analyze)
    monkeypatch.setattr(pipeline_runner, "cmd_score", fake_score)
    monkeypatch.setattr(pipeline_runner, "cmd_render", fake_render)
    return captured


def test_run_render_only_skips_analyze_and_score(tmp_path: Path, monkeypatch):
    """`run_render_only` must dispatch to cmd_render with the plan path,
    and never invoke analyze or score."""
    captured = _patch_pipeline(monkeypatch)

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"entries": []}))
    output = tmp_path / "out.mp4"

    # Pass reveal=False so we don't accidentally trigger macOS UI in CI.
    opts = AutoOpts(reveal=False)
    result = run_render_only(plan, output, opts)

    assert result == output.resolve()
    assert "analyze" not in captured, "analyze must not run on --from-plan"
    assert "score" not in captured, "score must not run on --from-plan"
    assert "render" in captured, "render must run on --from-plan"
    assert Path(captured["render"].plan) == plan.resolve()
    assert Path(captured["render"].output) == output.resolve()


def test_cli_auto_from_plan_routes_to_render_only(tmp_path: Path, monkeypatch):
    """`aftermovie auto --from-plan ... --output ...` parses and dispatches
    to `run_render_only` without requiring --clips or --song."""
    captured = _patch_pipeline(monkeypatch)
    seen: dict[str, Path] = {}

    def fake_run_render_only(plan: Path, output: Path,
                             opts: AutoOpts | None = None) -> Path:
        seen["plan"] = Path(plan)
        seen["output"] = Path(output)
        Path(output).write_bytes(b"")
        return Path(output).resolve()

    # cli.py imports run_render_only by name, so patch it on the cli module.
    monkeypatch.setattr(cli, "run_render_only", fake_run_render_only)

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"entries": []}))
    output = tmp_path / "out.mp4"

    parser = cli.build_parser()
    args = parser.parse_args([
        "auto", "--from-plan", str(plan), "--output", str(output),
        "--no-reveal",
    ])
    args.func(args)

    assert "analyze" not in captured, "analyze must not run on --from-plan"
    assert "score" not in captured, "score must not run on --from-plan"
    assert seen["plan"] == plan.resolve()
    assert seen["output"] == output.resolve()


def test_cli_render_from_plan_alias(tmp_path: Path, monkeypatch):
    """`aftermovie render-from-plan` is an alias for `render` — same handler.

    `cli.py` imports `cmd_render` directly from `aftermovie.render.pipeline`
    and wires it as the subparser's `func`, so the alias parser captures the
    *real* function reference at parser-build time. We assert the parser
    binds the same handler for both subcommands rather than stubbing.
    """
    from aftermovie.render.pipeline import cmd_render as real_cmd_render

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"entries": []}))
    output = tmp_path / "out.mp4"

    parser = cli.build_parser()
    args_alias = parser.parse_args([
        "render-from-plan", "--plan", str(plan), "--output", str(output),
    ])
    args_real = parser.parse_args([
        "render", "--plan", str(plan), "--output", str(output),
    ])
    assert args_alias.func is real_cmd_render
    assert args_alias.func is args_real.func, (
        "render-from-plan must dispatch to the same handler as render"
    )


def test_cli_auto_requires_clips_or_from_plan(monkeypatch):
    """Without --from-plan, --clips and --song are required."""
    _patch_pipeline(monkeypatch)
    parser = cli.build_parser()
    args = parser.parse_args(["auto"])
    with pytest.raises(SystemExit):
        args.func(args)

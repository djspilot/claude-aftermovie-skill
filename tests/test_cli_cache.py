"""Tests for `aftermovie cache stats` and `aftermovie cache clear` subcommands.

The CLI is a thin Adapter over `PrerenderCache`: stats reads its `stats()`
snapshot, clear calls `clear()` after a confirmation prompt unless `--yes`.
Both are exercised here against an isolated cache root.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aftermovie import config
from aftermovie.cli import build_parser
from aftermovie.render.prerender_cache import PrerenderCache


def _isolated_data_dir(tmp_path: Path, monkeypatch) -> Path:
    fake = tmp_path / "state"
    fake.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: fake)
    return fake


def _seed_cache_entry(cache: PrerenderCache, tmp_path: Path) -> Path:
    """Drop one real file into the cache so stats/clear have something to chew on."""
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"x" * 128)
    key = "c" * 40  # 40 hex chars — content irrelevant for this test
    return cache.put(key, rendered)


# ---- stats ----------------------------------------------------------------

def test_cache_stats_prints_expected_lines(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """`aftermovie cache stats` prints disk usage, hit/miss counts, etc."""
    _isolated_data_dir(tmp_path, monkeypatch)
    cache = PrerenderCache()
    _seed_cache_entry(cache, tmp_path)
    # Synthesise one hit + one miss so the hit/miss lines aren't both 0.
    cache.get("c" * 40)
    cache.get("not-a-real-key")

    parser = build_parser()
    args = parser.parse_args(["cache", "stats"])
    args.func(args)

    out = capsys.readouterr().out
    assert "aftermovie prerender cache" in out
    assert "entries:" in out
    assert "disk used:" in out
    assert "hits:" in out
    assert "misses:" in out
    assert "hit rate:" in out


def test_cache_stats_handles_empty_cache(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """`stats` on a never-used cache still prints sensible output, not a crash."""
    _isolated_data_dir(tmp_path, monkeypatch)
    parser = build_parser()
    args = parser.parse_args(["cache", "stats"])
    args.func(args)
    out = capsys.readouterr().out
    assert "entries:     0" in out
    assert "hits:        0" in out
    assert "misses:      0" in out


# ---- clear ----------------------------------------------------------------

def test_cache_clear_with_yes_empties_the_dir(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """`cache clear --yes` skips the prompt and wipes everything."""
    _isolated_data_dir(tmp_path, monkeypatch)
    cache = PrerenderCache()
    stored = _seed_cache_entry(cache, tmp_path)
    assert stored.is_file()
    root = cache.root
    assert root.is_dir()

    parser = build_parser()
    args = parser.parse_args(["cache", "clear", "--yes"])
    args.func(args)

    # The cache root is removed; future operations will recreate it.
    assert not root.exists()
    out = capsys.readouterr().out
    assert "Cleared" in out


def test_cache_clear_aborts_on_negative_prompt(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """Answering 'n' at the confirmation prompt leaves the cache intact."""
    _isolated_data_dir(tmp_path, monkeypatch)
    cache = PrerenderCache()
    stored = _seed_cache_entry(cache, tmp_path)
    assert stored.is_file()

    # Simulate the user saying "n".
    monkeypatch.setattr("builtins.input", lambda *_args, **_kw: "n")
    parser = build_parser()
    args = parser.parse_args(["cache", "clear"])
    args.func(args)

    assert stored.is_file(), "clear must abort on negative confirmation"
    out = capsys.readouterr().out
    assert "Aborted" in out

"""End-to-end tests for `aftermovie import` — the CLI Adapter over import_sources."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aftermovie import optional_dep
from aftermovie.cli import build_parser
from aftermovie.import_sources import gopro as gopro_mod
from aftermovie.import_sources import gopro_icc as icc_mod


@pytest.fixture(autouse=True)
def _clean_optional_dep_registry():
    optional_dep._reset_for_tests()
    yield
    optional_dep._reset_for_tests()


def _seed_gopro_dcim(root: Path) -> Path:
    mount = root / "GOPRO"
    dcim = mount / "DCIM" / "100GOPRO"
    dcim.mkdir(parents=True)
    now = datetime.now()
    for name in ("GX010001.MP4", "GOPR0002.JPG"):
        f = dcim / name
        f.write_bytes(b"\x00" * 64)
        os.utime(f, (now.timestamp(), now.timestamp()))
    return mount


def _patch_registry_with_single_gopro(monkeypatch, mount: Path) -> None:
    """Hide real /Volumes detection; expose only the fixture mount.

    Also pins the ICC GoPro branch to empty so a HERO9 happens to be
    plugged in on the dev machine doesn't bleed into CLI tests.
    """
    monkeypatch.setattr(gopro_mod, "detect_gopro_mounts", lambda: [mount])
    monkeypatch.setattr(icc_mod, "detect_icc_gopros", lambda: [])


def test_import_dry_run_writes_nothing(monkeypatch, tmp_path: Path, capsys):
    """`--dry-run` lists candidates but creates no dest files."""
    mount = _seed_gopro_dcim(tmp_path)
    _patch_registry_with_single_gopro(monkeypatch, mount)

    dest_parent = tmp_path / "out"
    parser = build_parser()
    args = parser.parse_args([
        "import",
        "--since", "2026-01-01",
        "--to", str(dest_parent),
        "--sources", "gopro",
        "--dry-run",
    ])
    args.func(args)

    # The dry-run path must not have created any subfolder or copied files.
    assert not dest_parent.exists() or not any(dest_parent.iterdir())
    err = capsys.readouterr().err
    assert "dry-run" in err


def test_import_creates_date_stamped_subfolder(monkeypatch, tmp_path: Path):
    """A real run materializes `<to>/YYYY-MM-DD_to_YYYY-MM-DD/` and copies."""
    mount = _seed_gopro_dcim(tmp_path)
    _patch_registry_with_single_gopro(monkeypatch, mount)

    dest_parent = tmp_path / "out"
    parser = build_parser()
    args = parser.parse_args([
        "import",
        "--since", "2026-01-01",
        "--until", "2099-12-31",
        "--to", str(dest_parent),
        "--sources", "gopro",
    ])
    args.func(args)

    # Exactly one subfolder shaped like "2026-01-01_to_2099-12-31".
    children = list(dest_parent.iterdir())
    assert len(children) == 1
    sub = children[0]
    assert sub.is_dir()
    assert sub.name == "2026-01-01_to_2099-12-31"
    # Both seeded files made it across.
    copied_names = {p.name for p in sub.iterdir()}
    assert copied_names == {"GX010001.MP4", "GOPR0002.JPG"}


def test_import_until_defaults_to_now(monkeypatch, tmp_path: Path):
    """Omitting --until uses `datetime.now()` for the upper bound."""
    mount = _seed_gopro_dcim(tmp_path)
    _patch_registry_with_single_gopro(monkeypatch, mount)

    dest_parent = tmp_path / "out"
    parser = build_parser()
    args = parser.parse_args([
        "import",
        "--since", "2026-01-01",
        "--to", str(dest_parent),
        "--sources", "gopro",
    ])
    args.func(args)
    today = datetime.now().strftime("%Y-%m-%d")
    sub_names = [p.name for p in dest_parent.iterdir()]
    assert len(sub_names) == 1
    assert sub_names[0].startswith("2026-01-01_to_")
    assert sub_names[0].endswith(today)


def test_import_rejects_invalid_since(monkeypatch, tmp_path: Path):
    parser = build_parser()
    args = parser.parse_args([
        "import",
        "--since", "not-a-date",
        "--to", str(tmp_path),
        "--sources", "gopro",
    ])
    with pytest.raises(SystemExit) as e:
        args.func(args)
    assert "--since" in str(e.value)


def test_import_rejects_unknown_source(monkeypatch, tmp_path: Path):
    parser = build_parser()
    args = parser.parse_args([
        "import",
        "--since", "2026-01-01",
        "--to", str(tmp_path),
        "--sources", "nope_not_real",
    ])
    with pytest.raises(SystemExit) as e:
        args.func(args)
    assert "No sources match" in str(e.value)


def test_import_idempotent_second_run(monkeypatch, tmp_path: Path):
    """Run twice → second pass copies 0 new files but creates no error."""
    mount = _seed_gopro_dcim(tmp_path)
    _patch_registry_with_single_gopro(monkeypatch, mount)

    dest_parent = tmp_path / "out"
    parser = build_parser()
    args = parser.parse_args([
        "import",
        "--since", "2026-01-01",
        "--to", str(dest_parent),
        "--sources", "gopro",
    ])
    args.func(args)
    args2 = parser.parse_args([
        "import",
        "--since", "2026-01-01",
        "--to", str(dest_parent),
        "--sources", "gopro",
    ])
    # Re-running must not raise and must not duplicate files.
    args2.func(args2)
    sub = next(dest_parent.iterdir())
    copied_names = sorted(p.name for p in sub.iterdir())
    assert copied_names == ["GOPR0002.JPG", "GX010001.MP4"]

"""Tests for Phase D — song picker in the GUI.

Covers the SelectionService Module extensions (`set_song`, `current_song`,
`list_candidate_songs`, `song_info`, `recent_songs`) and the HTTP Adapter
endpoints they back. The tests are unit-level against the service where
possible; only the path-validation 400 contract goes through the real
HTTP layer because that's where the bad-request → 400 mapping lives.

The recents sidecar lives at `~/.aftermovie/recent-songs.json`, so each
test that touches recents redirects `Path.home()` into `tmp_path` to
avoid clobbering the developer's real recents file.
"""
from __future__ import annotations

import json
import shutil
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from aftermovie.analyze.sidecar import clear_all_caches
from aftermovie.select.server import SelectServer
from aftermovie.select.service import (
    AUDIO_EXTS,
    RECENT_SONGS_CAP,
    RECENT_SONGS_FILENAME,
    SelectionService,
    _downsample_energy,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


# ---- helpers ---------------------------------------------------------------

def _seed_folder(tmp_path: Path, fixtures_dir: Path) -> Path:
    """Same shape as the other test files — mp4s + one still."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    shutil.copy(fixtures_dir / "clip_a.mp4", clips_dir / "clip_a.mp4")
    shutil.copy(fixtures_dir / "clip_b.mp4", clips_dir / "clip_b.mp4")
    Image.new("RGB", (200, 150), (180, 30, 30)).save(clips_dir / "still.jpg")
    return clips_dir


def _redirect_home(monkeypatch, tmp_path: Path) -> Path:
    """Redirect `Path.home()` into `tmp_path / "home"` so the recents sidecar
    lands in a hermetic dir per-test instead of `~/.aftermovie`."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _redirect_data_dir(monkeypatch, tmp_path: Path) -> Path:
    """Redirect `aftermovie.config.data_dir()` so song-analysis cache lands
    in a hermetic dir per-test instead of `~/.skills-data/aftermovie/`."""
    from aftermovie import config
    fake = tmp_path / "skills-data"
    fake.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "data_dir", lambda: fake)
    return fake


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_post(url: str, body: dict, timeout: float = 5.0) -> tuple[int, bytes]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


# ---- AUDIO_EXTS sanity check ----------------------------------------------

def test_audio_exts_lists_documented_extensions() -> None:
    """The picker covers the six extensions called out in the design doc."""
    assert {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"} <= AUDIO_EXTS


# ---- set_song + current_song ----------------------------------------------

def test_set_song_updates_current_song_and_recents(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """`set_song` flips `current_song` AND prepends to the recents sidecar."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)
    assert svc.current_song() == {"path": None}

    song = fixtures_dir / "tone.wav"
    info = svc.set_song(song)
    assert info["path"] == str(song.resolve())
    assert info["name"] == "tone.wav"

    # current_song reflects the new state (without forcing librosa).
    current = svc.current_song()
    assert current["path"] == str(song.resolve())
    assert current["name"] == "tone.wav"

    # Recents got the entry. last_used is a fresh epoch.
    rec = svc.recent_songs()
    assert len(rec) == 1
    assert rec[0]["path"] == str(song.resolve())
    assert rec[0]["last_used"] > 0


def test_set_song_rejects_missing_file(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """A non-existent path raises ValueError (HTTP Adapter maps to 400)."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)
    with pytest.raises(ValueError):
        svc.set_song(tmp_path / "no_such_song.mp3")


def test_set_song_rejects_non_audio_extension(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """A file with the wrong extension is rejected before activation."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)
    with pytest.raises(ValueError):
        svc.set_song(clips_dir / "clip_a.mp4")  # video, not audio


# ---- list_candidate_songs --------------------------------------------------

def test_list_candidate_songs_finds_audio_excludes_video(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """Audio files are returned; mp4 / still files are not."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    # Drop a real audio file into the clips folder.
    shutil.copy(fixtures_dir / "tone.wav", clips_dir / "soundtrack.wav")
    # And a non-audio file we expect to NOT show up.
    (clips_dir / "notes.txt").write_text("nope")

    svc = SelectionService(clips_dir)
    rows = svc.list_candidate_songs()
    names = {r["name"] for r in rows}
    assert "soundtrack.wav" in names, (
        f"expected soundtrack.wav in candidates, got {names}"
    )
    assert "clip_a.mp4" not in names
    assert "notes.txt" not in names
    # duration_s is None when not yet analyzed (we never block on librosa here).
    for r in rows:
        assert "duration_s" in r


# ---- song_info caching -----------------------------------------------------

def test_song_info_caches_second_call(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """Second `song_info(path)` reads from disk; no re-analyze."""
    _redirect_home(monkeypatch, tmp_path)
    cache_root = _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    song = fixtures_dir / "tone.wav"
    svc = SelectionService(clips_dir)

    # Stub `analyze_song` so we can count invocations without paying librosa.
    calls = {"n": 0}

    def fake_analyze(p):
        calls["n"] += 1
        return {
            "duration_s": 3.0,
            "tempo_bpm": 120.0,
            "energy_per_s": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

    import aftermovie.score.song as song_mod
    monkeypatch.setattr(song_mod, "analyze_song", fake_analyze)

    first = svc.song_info(song)
    assert first["duration_s"] == pytest.approx(3.0)
    assert first["tempo_bpm"] == pytest.approx(120.0)
    assert len(first["energy_curve_samples"]) >= 1
    assert calls["n"] == 1

    # Cache file landed on disk under the redirected data_dir.
    cache_dir = cache_root / "song-analysis"
    assert cache_dir.is_dir()
    assert any(cache_dir.glob("*.json"))

    # Second call reads the cache — no re-analyze.
    second = svc.song_info(song)
    assert second["duration_s"] == pytest.approx(3.0)
    assert calls["n"] == 1, "expected the second song_info call to hit the cache"


def test_song_info_rejects_missing_path(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """A missing path raises ValueError (HTTP Adapter maps to 400)."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    svc = SelectionService(clips_dir)
    with pytest.raises(ValueError):
        svc.song_info(tmp_path / "missing.mp3")


# ---- recents LRU ----------------------------------------------------------

def test_recent_songs_caps_at_ten_with_lru(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """Pushing > 10 songs evicts the oldest; the head is most-recent."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    songs_dir = tmp_path / "songs"
    songs_dir.mkdir()
    # Seed 12 distinct WAVs by copying the fixture under unique names.
    paths: list[Path] = []
    src = fixtures_dir / "tone.wav"
    for i in range(12):
        p = songs_dir / f"song_{i:02d}.wav"
        shutil.copy(src, p)
        paths.append(p)

    svc = SelectionService(clips_dir)
    for p in paths:
        svc.set_song(p)

    rec = svc.recent_songs()
    assert len(rec) == RECENT_SONGS_CAP == 10
    # Head is the most-recently set (paths[-1]); tail dropped the oldest two.
    assert rec[0]["path"] == str(paths[-1].resolve())
    head_paths = {r["path"] for r in rec}
    assert str(paths[0].resolve()) not in head_paths
    assert str(paths[1].resolve()) not in head_paths

    # Re-picking an existing entry moves it back to the head without dupes.
    svc.set_song(paths[5])
    rec2 = svc.recent_songs()
    assert rec2[0]["path"] == str(paths[5].resolve())
    paths_in_list = [r["path"] for r in rec2]
    assert paths_in_list.count(str(paths[5].resolve())) == 1


def test_recent_songs_sidecar_filename_matches_spec(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """The recents file lives at `~/.aftermovie/recent-songs.json`."""
    home = _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    song = fixtures_dir / "tone.wav"
    svc = SelectionService(clips_dir)
    svc.set_song(song)

    sidecar = home / ".aftermovie" / RECENT_SONGS_FILENAME
    assert sidecar.is_file(), f"expected recents sidecar at {sidecar}"
    payload = json.loads(sidecar.read_text())
    assert payload["version"] == 1
    assert isinstance(payload["songs"], list)
    assert payload["songs"][0]["path"] == str(song.resolve())


# ---- _downsample_energy ----------------------------------------------------

def test_downsample_energy_targets_n_buckets() -> None:
    """The sparkline helper hits the requested bucket count + clamps to [0, 1]."""
    out = _downsample_energy([0.0, 0.5, 1.0] * 100, n=50)
    assert len(out) == 50
    assert all(0.0 <= v <= 1.0 for v in out)

    # Empty / zero-bucket input degrades gracefully.
    assert _downsample_energy([], n=50) == []
    assert _downsample_energy([0.5], n=0) == []
    # Pass-through when shorter than target.
    assert _downsample_energy([0.1, 0.2, 0.3], n=50) == [0.1, 0.2, 0.3]


# ---- HTTP layer: /api/set-song-path 200 + 400 ------------------------------

def test_post_set_song_path_accepts_valid_audio(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """POST /api/set-song-path with a valid audio file returns ok + path."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    port = _free_port()
    song = fixtures_dir / "tone.wav"
    with SelectServer(clips_dir, port=port) as srv:
        status, body = _http_post(
            f"{srv.url}api/set-song-path", {"path": str(song)},
        )
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["path"] == str(song.resolve())


def test_post_set_song_path_rejects_missing_file(
    tmp_path: Path, fixtures_dir: Path, monkeypatch,
) -> None:
    """POST /api/set-song-path with a missing path → HTTP 400 bad_request."""
    _redirect_home(monkeypatch, tmp_path)
    _redirect_data_dir(monkeypatch, tmp_path)
    clear_all_caches()

    clips_dir = _seed_folder(tmp_path, fixtures_dir)
    port = _free_port()
    with SelectServer(clips_dir, port=port) as srv:
        try:
            _http_post(
                f"{srv.url}api/set-song-path",
                {"path": str(tmp_path / "no_such_song.mp3")},
            )
            raise AssertionError(
                "expected 400 from /api/set-song-path with a missing file",
            )
        except urllib.error.HTTPError as e:
            assert e.code == 400
            payload = json.loads(e.read())
            assert payload["error"] == "bad_request"

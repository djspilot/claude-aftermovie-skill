"""MCP server tests — drive the FastMCP tools via an in-process client session."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


skip_no_mcp = pytest.mark.skipif(not _try_import("mcp"), reason="mcp not installed")
skip_no_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


async def _run_tool(client, name: str, args: dict) -> dict:
    result = await client.call_tool(name, args)
    # FastMCP wraps the return value; structured_content is the dict.
    if result.structuredContent is not None:
        return result.structuredContent
    # Fallback: parse the first text content block.
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    return {}


def _client_session(monkeypatch_data_dir: Path):
    from mcp.shared.memory import create_connected_server_and_client_session
    from aftermovie.mcp_server.server import mcp
    return create_connected_server_and_client_session(mcp)


@skip_no_mcp
def test_list_themes_returns_four_themes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    async def go():
        async with _client_session(tmp_path) as client:
            await client.initialize()
            result = await _run_tool(client, "list_themes", {})
            names = [t["name"] for t in result["themes"]]
            assert set(names) == {"cinematic", "punchy", "chill", "nostalgic"}

    asyncio.run(go())


@skip_no_mcp
@skip_no_ffmpeg
def test_full_pipeline_via_mcp(tmp_path: Path, fixtures_dir: Path, tone: Path, monkeypatch):
    """analyze → propose → render → poll, all over the MCP wire."""
    monkeypatch.setenv("HOME", str(tmp_path))

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("clip_a.mp4", "clip_b.mp4"):
        shutil.copy(fixtures_dir / name, clips_dir / name)

    out_mp4 = tmp_path / "mcp_render.mp4"

    async def go():
        async with _client_session(tmp_path) as client:
            await client.initialize()

            r = await _run_tool(client, "analyze_folder", {"path": str(clips_dir)})
            catalog_id = r["catalog_id"]
            job_id = r["job_id"]
            assert catalog_id

            # If a job was started, poll until done.
            if job_id:
                for _ in range(60):
                    status = await _run_tool(client, "get_job", {"job_id": job_id})
                    if status["status"] in ("done", "error", "cancelled"):
                        break
                    await asyncio.sleep(0.5)
                assert status["status"] == "done", f"analyze failed: {status}"

            r = await _run_tool(client, "propose_plan", {
                "catalog_id": catalog_id,
                "song_path": str(tone),
                "theme": "cinematic",
                "target_length_s": 5,
                "aspect": "16:9",
            })
            plan_id = r["plan_id"]
            assert r["summary"]["n_cuts"] >= 1

            plan = await _run_tool(client, "get_plan", {"plan_id": plan_id})
            assert plan["theme"] == "cinematic"
            assert plan["target_length_s"] <= 5.01

            r = await _run_tool(client, "render_plan", {
                "plan_id": plan_id, "output_path": str(out_mp4),
            })
            job_id = r["job_id"]
            for _ in range(120):
                status = await _run_tool(client, "get_job", {"job_id": job_id})
                if status["status"] in ("done", "error", "cancelled"):
                    break
                await asyncio.sleep(0.5)
            assert status["status"] == "done", f"render failed: {status}"
            assert out_mp4.exists()

    asyncio.run(go())


@skip_no_mcp
def test_tweak_plan_exclude_source(tmp_path: Path, monkeypatch):
    """Build a plan in-memory and apply a tweak via MCP."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from aftermovie.state import save_plan

    plan = {
        "plan_id": "abc123",
        "catalog_id": "x",
        "entries": [
            {"source": "/a.mp4", "start_s": 0, "end_s": 2, "out_duration_s": 2,
             "speed": 1.0, "beat_time_s": 0.0, "score": 5.0, "reasons": []},
            {"source": "/b.mp4", "start_s": 0, "end_s": 2, "out_duration_s": 2,
             "speed": 1.0, "beat_time_s": 2.0, "score": 4.0, "reasons": []},
            {"source": "/a.mp4", "start_s": 2, "end_s": 4, "out_duration_s": 2,
             "speed": 1.0, "beat_time_s": 4.0, "score": 3.0, "reasons": []},
        ],
        "music_db": -8.0,
        "aspect": "16:9",
    }
    save_plan("abc123", plan)

    async def go():
        async with _client_session(tmp_path) as client:
            await client.initialize()

            r = await _run_tool(client, "tweak_plan", {
                "plan_id": "abc123",
                "ops": [{"op": "exclude_source", "source": "/a.mp4"}],
            })
            new_id = r["plan_id"]
            new_plan = await _run_tool(client, "get_plan", {"plan_id": new_id})
            sources = {e["source"] for e in new_plan["entries"]}
            assert sources == {"/b.mp4"}, f"unexpected sources: {sources}"

            r = await _run_tool(client, "tweak_plan", {
                "plan_id": new_id,
                "ops": [{"op": "set", "path": "music_db", "value": -12.0}],
            })
            new_id2 = r["plan_id"]
            new_plan2 = await _run_tool(client, "get_plan", {"plan_id": new_id2})
            assert new_plan2["music_db"] == -12.0

    asyncio.run(go())

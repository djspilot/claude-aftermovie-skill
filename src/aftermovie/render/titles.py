"""Title card and lower-third generation.

ffmpeg's drawtext filter requires a libfreetype-enabled build, which homebrew's
default `ffmpeg` formula does NOT include. Instead we render titles to RGBA
PNGs via PIL and ask ffmpeg to overlay them — guaranteed portable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from aftermovie.config import fonts_dir

# Per-theme typography. font_kind selects which file in assets/fonts to prefer.
_THEME_TYPO = {
    "cinematic": {"font_kind": "display", "size_frac": 0.07, "color": (255, 255, 255, 255),
                  "shadow": (0, 0, 0, 160), "tracking": 0.04, "align": "center"},
    "punchy":    {"font_kind": "display", "size_frac": 0.11, "color": (255, 255, 255, 255),
                  "shadow": (0, 0, 0, 200), "tracking": 0.06, "align": "center", "upper": True},
    "chill":     {"font_kind": "sans",    "size_frac": 0.06, "color": (255, 255, 255, 220),
                  "shadow": None, "tracking": 0.02, "align": "center"},
    "nostalgic": {"font_kind": "display", "size_frac": 0.08, "color": (245, 230, 197, 255),
                  "shadow": (60, 30, 0, 180), "tracking": 0.05, "align": "center"},
}

_SYS_FALLBACKS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Avenir.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def _resolve_font_path(kind: str) -> str | None:
    d = fonts_dir()
    if d.is_dir():
        for pattern in (f"*{kind}*.ttf", f"*{kind}*.otf", "*.ttf", "*.otf"):
            for f in sorted(d.glob(pattern)):
                return str(f)
    for p in _SYS_FALLBACKS:
        if Path(p).is_file():
            return p
    return None


def render_title_png(text: str, theme: str, frame_w: int, frame_h: int,
                     out_path: Path) -> Path:
    """Render a transparent RGBA PNG with the title text centered."""
    typo = _THEME_TYPO.get(theme, _THEME_TYPO["cinematic"])
    display_text = text.upper() if typo.get("upper") else text

    font_size = max(20, int(frame_h * typo["size_frac"]))
    font_path = _resolve_font_path(typo["font_kind"])
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except (OSError, ValueError):
        font = ImageFont.load_default()

    img = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Measure text bbox.
    try:
        l, t, r, b = draw.textbbox((0, 0), display_text, font=font)
        tw, th = r - l, b - t
        offset_x, offset_y = -l, -t
    except AttributeError:
        tw, th = draw.textsize(display_text, font=font)
        offset_x = offset_y = 0

    x = (frame_w - tw) // 2 + offset_x
    y = (frame_h - th) // 2 + offset_y

    shadow = typo.get("shadow")
    if shadow:
        draw.text((x + 3, y + 3), display_text, font=font, fill=shadow)
    draw.text((x, y), display_text, font=font, fill=typo["color"])

    img.save(out_path)
    return out_path


def build_overlay_chain(input_label: str, title_inputs: list[tuple[int, dict]],
                        out_label: str = "v_titled") -> str:
    """
    Build the filter_complex segment that overlays each title PNG on top of
    `input_label` and produces `out_label`.

    title_inputs is a list of (ffmpeg_input_index, title_dict). Each title
    dict contains kind, text, start_s, duration_s.
    """
    if not title_inputs:
        return ""
    parts = []
    cur = input_label
    for n, (idx, t) in enumerate(title_inputs):
        start = t["start_s"]
        end = start + t["duration_s"]
        next_label = f"v_t{n}" if n < len(title_inputs) - 1 else out_label
        enable = f"between(t,{start:.3f},{end:.3f})"
        parts.append(
            f"[{cur}][{idx}:v]overlay=enable='{enable}':x=0:y=0[{next_label}]"
        )
        cur = next_label
    return ";".join(parts)


def resolve_title_times(titles: list[dict[str, Any]],
                        total_duration_s: float) -> list[dict[str, Any]]:
    """Fill in `start_s` on each title based on its `kind`."""
    out: list[dict[str, Any]] = []
    for t in titles:
        kind = t.get("kind", "intro")
        duration = float(t.get("duration_s", 2.0))
        if kind == "intro":
            start = 0.0
        elif kind == "outro":
            start = max(0.0, total_duration_s - duration)
        else:
            start = float(t.get("at_s", 0.0) or 0.0)
        out.append({
            "kind": kind,
            "text": t.get("text", ""),
            "start_s": start,
            "duration_s": duration,
        })
    return out

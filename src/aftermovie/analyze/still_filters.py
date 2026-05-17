"""Still-image filter compiler — picks a Quik-style display variant per file
and builds the matching ffmpeg `filter_complex` chain.

Split out of `stills.py` so the discovery / IO half of that module stays
under 200 LOC. The variant picker is deterministic per filename (hash of
the file stem), so the same photo always animates the same way across runs.

Variants
--------
    live       — subtle 1.00 → 1.05 zoom-in (the original Ken Burns default)
    push       — 1.00 → 1.10 push-in
    pull       — 1.10 → 1.00 push-out
    pan_h      — horizontal pan across the frame, no zoom change
    fit_pad    — `letterbox` the image; leaves visible black bars
    blurred_bg — image fit on top of a blurred copy of itself (no bars)
    shake      — subtle handheld jitter overlay

Portrait sources can't safely use the crop-heavy variants without lopping
off heads, so `_pick_still_variant` gates them to fit_pad / blurred_bg.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# All variants that crop / zoom into the source. Narrow-aspect images
# shouldn't land in this set; the picker swaps them for a "show the whole
# image" variant so heads / feet don't get cropped off.
_FILL_VARIANTS = ("live", "push", "pull", "pan_h", "shake")

# Variants that show the entire source image (no cropping). Narrow-aspect
# sources are forced into this set. User wanted literal black borders only —
# blurred_bg was confusing because the blurred-self bg still reads as zoomed.
_PORTRAIT_VARIANTS = ("fit_pad",)

# Full ordered tuple for general (landscape / square-ish) sources. live is
# weighted by listing it twice so the historical look stays the most common.
_ALL_VARIANTS = ("live", "live", "push", "pull", "pan_h", "fit_pad", "blurred_bg", "shake")

# Fractional threshold: if source aspect / target aspect is below this, we
# treat the source as "narrow" and gate it to _PORTRAIT_VARIANTS regardless
# of whether it's strictly h > w. Catches 4:3 stills on 16:9 output too.
_NARROW_ASPECT_RATIO = 0.85


@dataclass(frozen=True)
class FilterSpec:
    """Compiled filter_complex fragment for a still clip.

    Attributes
    ----------
    chain : str
        Body of the filter chain — does NOT include the trailing label.
        Caller wraps it with `[in]<chain>[out_label]` form.
    fade_d : str
        Suggested fade duration string (e.g. "0.20"). May be empty.
    out_label : str
        Suggested unique output label for use in a filter_complex graph.
    """

    chain: str
    fade_d: str
    out_label: str


def _image_dims(path: Path) -> tuple[int, int] | None:
    """Return (width, height) as the image *displays*, respecting EXIF orientation.

    iPhone photos are stored with raw sensor dimensions (e.g. 4032×3024) plus
    an EXIF Orientation tag that flips them on display. Without
    `exif_transpose` a vertical iPhone photo reads as landscape here and
    skips the letterbox path.
    """
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:  # noqa: BLE001 — defensive: any malformed exif
                pass
            return img.size
    except (OSError, ValueError, ImportError):
        return None


def _is_portrait(path: Path) -> bool:
    """True if the underlying image is taller than wide.

    Kept for backwards-compat with tests / callers. Prefer `_should_letterbox`
    which also catches near-square sources on widescreen output.
    """
    dims = _image_dims(path)
    if dims is None:
        return False
    return dims[1] > dims[0]


def _should_letterbox(path: Path, target_w: int, target_h: int) -> bool:
    """True if the source is meaningfully narrower than the target frame.

    Filling a 9:10 portrait into a 16:9 frame chops heads off; filling a
    4:3 photo into 16:9 still loses noticeable top-and-bottom content. Both
    cases benefit from letterboxing instead of cropping.
    """
    dims = _image_dims(path)
    if dims is None:
        return False
    src_w, src_h = dims
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return False
    return (src_w / src_h) < (target_w / target_h) * _NARROW_ASPECT_RATIO


def _stem_seed(path: Path) -> int:
    """Deterministic 32-bit integer seed from the file stem.

    Using the stem (not the full path) means the same photo gets the same
    variant whether it's referenced from the source folder or the cache.
    """
    h = hashlib.sha1(path.stem.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _pick_still_variant(path: Path, target_w: int, target_h: int) -> tuple[str, int]:
    """Return (variant_name, seed) for a still.

    Deterministic — same filename → same variant on every run.
    Narrow-aspect sources (portrait, near-portrait, taller-than-target)
    are gated to _PORTRAIT_VARIANTS so we never crop a head off and we
    favour literal letterbox (`fit_pad`) over the blurred-self fill.
    """
    seed = _stem_seed(path)
    if _should_letterbox(path, target_w, target_h):
        choices = _PORTRAIT_VARIANTS
    else:
        choices = _ALL_VARIANTS
    variant = choices[seed % len(choices)]
    return variant, seed


def _zoompan_filter(z_expr: str, x_expr: str, y_expr: str,
                    n: int, fps: int, w: int, h: int) -> str:
    """Build a zoompan fragment with custom z/x/y expressions.

    The 2× supersample → crop → zoompan idiom is shared by every cropping
    variant: it lets us pan/zoom without blowing up jaggies on the input.
    """
    return (
        f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
        f"crop={w*2}:{h*2},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={n}:fps={fps}:s={w}x{h},"
        f"format=yuv420p"
    )


def _build_still_filter(variant: str, seed: int, n: int, fps: int,
                         w: int, h: int) -> FilterSpec:
    """Compile the ffmpeg filter chain for the chosen variant.

    Returns a FilterSpec with `.chain`, `.fade_d`, `.out_label` attributes.
    The chain doesn't include the surrounding `[in]…[out]` brackets — callers
    splice it into a larger filter_complex graph (or use it as `-vf`).
    """
    out_label = f"still_{seed:08x}"
    fade_d = "0.20"

    if variant == "live":
        # Historical default: zoom from 1.00 → 1.05, centered.
        chain = _zoompan_filter(
            z_expr=f"1+0.05*on/{n}",
            x_expr="iw/2-(iw/zoom/2)",
            y_expr="ih/2-(ih/zoom/2)",
            n=n, fps=fps, w=w, h=h,
        )
    elif variant == "push":
        chain = _zoompan_filter(
            z_expr=f"1+0.10*on/{n}",
            x_expr="iw/2-(iw/zoom/2)",
            y_expr="ih/2-(ih/zoom/2)",
            n=n, fps=fps, w=w, h=h,
        )
    elif variant == "pull":
        # Start zoomed in, end at 1.0.
        chain = _zoompan_filter(
            z_expr=f"1.10-0.10*on/{n}",
            x_expr="iw/2-(iw/zoom/2)",
            y_expr="ih/2-(ih/zoom/2)",
            n=n, fps=fps, w=w, h=h,
        )
    elif variant == "pan_h":
        # Horizontal pan: choose direction from seed parity.
        if seed % 2 == 0:
            x_expr = f"(iw-iw/zoom)*on/{n}"   # left → right
        else:
            x_expr = f"(iw-iw/zoom)*(1-on/{n})"  # right → left
        chain = _zoompan_filter(
            z_expr="1.10",  # fixed slight zoom so there's room to pan
            x_expr=x_expr,
            y_expr="ih/2-(ih/zoom/2)",
            n=n, fps=fps, w=w, h=h,
        )
    elif variant == "shake":
        # Subtle handheld jitter: small sinusoidal offsets on x/y.
        chain = _zoompan_filter(
            z_expr="1.06",
            x_expr=f"(iw-iw/zoom)/2 + 3*sin(2*PI*on/{max(1, n//6)})",
            y_expr=f"(ih-ih/zoom)/2 + 3*cos(2*PI*on/{max(1, n//7)})",
            n=n, fps=fps, w=w, h=h,
        )
    elif variant == "fit_pad":
        # Letterbox: scale to fit, pad with black.
        chain = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"format=yuv420p"
        )
    elif variant == "blurred_bg":
        # Fit on top of a blurred copy of itself — no black bars.
        chain = (
            f"split=2[bg][fg];"
            f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},boxblur=20:2[bgb];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,format=yuv420p"
        )
    else:
        # Defensive default — fall back to the historical "live" zoom.
        chain = _zoompan_filter(
            z_expr=f"1+0.05*on/{n}",
            x_expr="iw/2-(iw/zoom/2)",
            y_expr="ih/2-(ih/zoom/2)",
            n=n, fps=fps, w=w, h=h,
        )

    return FilterSpec(chain=chain, fade_d=fade_d, out_label=out_label)

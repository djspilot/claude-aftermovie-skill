"""Theme Resolver — single source of truth for theme bundle expansion.

A **Theme** is a preset bundle of look-and-feel knobs (LUT, music_db, pace,
transitions, audio_mix). The expansion rule "theme value wins over the
built-in default but loses to anything the user explicitly set" needs to be
identical whether you reach it through the CLI (`aftermovie auto --theme`),
the env file (`AFTERMOVIE_THEME=...`), or the MCP `auto(theme=...)` tool.

This Module owns:

* `THEMES` — the preset dict (re-exported from `aftermovie.config` for
  back-compat; the data itself lives here now).
* `ThemeResolver.apply(theme_name, current_values, defaults)` — the
  precedence kernel both `pipeline_runner._apply_theme` and
  `effective_config._theme_layer` call into.
* `ThemeResolver.describe(theme_name)` — pretty-printable view (description
  + theme-controlled knobs only, no `description` key bleed) used by
  `cmd_show_config` and `list_themes`.

Contract reminder: theme presets carry a `description` field used for help
text only — it must NEVER end up in the resolved knob values.
"""
from __future__ import annotations

from typing import Any, Mapping

# Local mirror of `aftermovie.config.DEFAULT_MUSIC_DB`. Duplicated (not
# imported) to avoid a circular import — `config.py` re-exports `THEMES`
# from this Module. Kept as a module-level constant so the duplication is
# loud and easy to grep if the canonical value ever moves.
_BASELINE_MUSIC_DB = -8.0

# ---- Theme presets ---------------------------------------------------------

# The single canonical theme dict. `aftermovie.config.THEMES` re-exports this
# so existing import sites keep working without churn.
THEMES: dict[str, dict] = {
    "cinematic": {
        "lut": "cinematic", "music_db": -9.0, "no_speed_ramp": False,
        "transitions": "soft", "audio_mix": "ducked", "pace": "medium",
        "description": "Glide-y crossfades, ducked audio — looks like a Quik edit.",
    },
    "punchy": {
        "lut": "punchy", "music_db": -6.0, "no_speed_ramp": False,
        "transitions": "auto", "audio_mix": "ducked", "pace": "fast",
        "description": "Fast cuts, whips on peaks, hot music — hype mode.",
    },
    "chill": {
        "lut": "chill", "music_db": -10.0, "no_speed_ramp": True,
        "transitions": "soft", "audio_mix": "ducked", "pace": "slow",
        "description": "Slow downbeat pacing, soft crossfades, no ramps.",
    },
    "nostalgic": {
        "lut": "nostalgic", "music_db": -10.0, "no_speed_ramp": False,
        "transitions": "soft", "audio_mix": "ducked", "pace": "medium",
        "description": "Film-look LUT, warm fades, ducked voice-aware audio.",
    },
}


# Snapshot of what each theme-controlled knob looks like at the built-in
# baseline. Used by callers (notably `pipeline_runner`) that work with a
# mutable dataclass and need to detect "is this field still at default?"
# before letting the theme bundle override it.
BASELINE_DEFAULTS: dict[str, Any] = {
    "lut": None,
    "music_db": _BASELINE_MUSIC_DB,
    "no_speed_ramp": False,
    "transitions": "cut",
    "audio_mix": "ducked",
    "pace": "medium",
}

# Metadata-only keys that must never escape into the resolved knob values.
_META_KEYS = frozenset({"description"})


class ThemeResolver:
    """The Module that owns theme expansion. Stateless — methods are
    classmethods so both surfaces (CLI/pipeline + MCP/effective_config)
    can call without instantiating.
    """

    @classmethod
    def apply(
        cls,
        theme_name: str | None,
        current_values: Mapping[str, Any],
        defaults: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the theme-overlaid values for `current_values`.

        Precedence (low -> high): `defaults` < theme preset < explicit
        non-default entries in `current_values`. Concretely, for each
        theme-controlled knob `k`:

        * If `current_values[k] == defaults.get(k)` the field is still at
          the baseline, so the theme's value wins.
        * Otherwise the caller set it explicitly — keep `current_values[k]`.

        Unknown / empty `theme_name` is a silent no-op (returns a copy of
        `current_values`). Theme metadata keys (`description`) are never
        copied through. Keys the theme mentions but `defaults` does not are
        skipped — the caller's value-set is the authority on what's
        addressable.
        """
        out: dict[str, Any] = dict(current_values)
        if not theme_name:
            return out
        preset = THEMES.get(theme_name)
        if not preset:
            return out
        for k, v in preset.items():
            if k in _META_KEYS:
                continue
            if k not in current_values:
                continue
            cur = current_values[k]
            baseline = defaults.get(k)
            if cur == baseline:
                out[k] = v
        return out

    @classmethod
    def describe(cls, theme_name: str) -> dict[str, Any]:
        """Return a display-shaped view of a theme: every theme-controlled
        knob plus the human description. Returns `{}` for unknown themes
        (matches the silent-no-op contract of `apply`)."""
        preset = THEMES.get(theme_name)
        if not preset:
            return {}
        knobs = {k: v for k, v in preset.items() if k not in _META_KEYS}
        return {
            "name": theme_name,
            "description": preset.get("description", ""),
            "values": knobs,
        }

    @classmethod
    def names(cls) -> list[str]:
        """Sorted list of known theme names (for `--theme` choices etc.)."""
        return sorted(THEMES.keys())

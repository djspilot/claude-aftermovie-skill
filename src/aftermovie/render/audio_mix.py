"""Final-mux audio filtergraphs for the three audio_mix modes.

Mode meanings:
    music_only — original behavior; drop clip audio, keep music at music_db.
    clip_only  — drop music, keep clip audio normalized.
    ducked     — mix music + clip audio; sidechain-compress music when the
                 clip-audio voice band is loud, so voices remain audible.
"""
from __future__ import annotations


def filtergraph(mode: str, music_db: float) -> str:
    """Return the -filter_complex string for the final mux step.

    Inputs assumed in the order: [0:a] = concatenated clip audio, [1:a] = music.
    Output label is [a_out].
    """
    if mode == "music_only":
        return f"[1:a]volume={music_db}dB[a_out]"

    if mode == "clip_only":
        return "[0:a]aresample=48000:async=1000[a_out]"

    if mode == "ducked":
        # Smoothed, low-pumping duck:
        #   - dynaudnorm flattens trigger so brief silences don't pump the music
        #   - longer attack/release so the duck glides instead of slamming
        #   - amix normalize=0 keeps the music_db level deterministic
        #   - final dynaudnorm evens overall loudness across cuts
        return (
            "[0:a]aresample=48000:async=1000,asetpts=N/SR/TB[clip];"
            "[clip]asplit=2[clip_out][clip_key];"
            "[clip_key]highpass=f=200,lowpass=f=3000,"
            "dynaudnorm=f=200:g=15:p=0.7,"
            "acompressor=threshold=-30dB:ratio=4:attack=20:release=400[trigger];"
            f"[1:a]volume={music_db}dB[m];"
            "[m][trigger]sidechaincompress=threshold=-26dB:ratio=3:"
            "attack=60:release=1200:makeup=1[m_ducked];"
            "[m_ducked][clip_out]amix=inputs=2:duration=first:normalize=0:"
            "weights='1 0.7'[a_mix];"
            "[a_mix]dynaudnorm=f=250:g=15:p=0.85[a_out]"
        )

    raise ValueError(f"unknown audio_mix mode: {mode}")


def needs_clip_audio(mode: str) -> bool:
    """True if the per-cut pre-render must keep clip audio (not -an)."""
    return mode in ("ducked", "clip_only")

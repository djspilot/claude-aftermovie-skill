"""aftermovie — GoPro-style beat-synced aftermovie generator."""
import os as _os

# scipy>=1.17's vendored ducc FFT backend spins up a thread pool sized by
# OMP_NUM_THREADS and registers an atfork handler that JOINS those workers
# on every fork. On macOS that join deadlocks the whole process the moment
# we subprocess ffmpeg after librosa has touched scipy.fft (observed: GUI
# render hanging forever in `libSystem_atfork_prepare → ducc0 → thread::join`).
# One FFT thread → empty pool → nothing to join. Our parallelism comes from
# worker PROCESSES (analyze pool, prerender pool), so this costs ~nothing.
# Respect an explicit user override; only set the default.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.2.0"

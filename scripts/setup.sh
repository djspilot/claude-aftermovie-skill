#!/usr/bin/env bash
# aftermovie/scripts/setup.sh — one-time setup, idempotent
# Installs Python deps into a local venv so the user's global Python stays clean.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${HOME}/.skills-data/aftermovie"
VENV="${DATA_DIR}/venv"

echo "==> aftermovie setup"
echo "    Skill:    ${SKILL_DIR}"
echo "    Data:     ${DATA_DIR}"

# Check ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ""
  echo "  ✗ ffmpeg not found. Install it first:"
  echo "      brew install ffmpeg"
  echo ""
  exit 1
fi
FFMPEG_VERSION="$(ffmpeg -version | head -n1)"
echo "    ffmpeg:   ${FFMPEG_VERSION}"

# Check Python ≥ 3.10
if ! command -v python3 >/dev/null 2>&1; then
  echo "  ✗ python3 not found. Install with: brew install python@3.12"
  exit 1
fi
PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "  ✗ Python ${PY_VERSION} is too old. Need ≥ 3.10. Install with: brew install python@3.12"
  exit 1
fi
echo "    python3:  ${PY_VERSION}"

# Create venv
mkdir -p "${DATA_DIR}"
if [ ! -d "${VENV}" ]; then
  echo "    Creating venv at ${VENV}..."
  python3 -m venv "${VENV}"
fi

# Install deps
echo "    Installing Python dependencies..."
"${VENV}/bin/pip" install --upgrade pip --quiet
"${VENV}/bin/pip" install --quiet \
  "librosa>=0.10" \
  "numpy>=1.24,<3" \
  "soundfile>=0.12" \
  "scipy>=1.10" \
  "tqdm>=4.65"

# Record the venv path for the CLI to find
cat > "${DATA_DIR}/config.json" <<EOF
{
  "venv": "${VENV}",
  "skill_dir": "${SKILL_DIR}",
  "data_dir": "${DATA_DIR}"
}
EOF

echo ""
echo "✓ Setup complete."
echo ""
echo "Quick test:"
echo "  python3 ${SKILL_DIR}/scripts/aftermovie.py --help"
echo ""
echo "Make a video:"
echo "  python3 ${SKILL_DIR}/scripts/aftermovie.py auto \\"
echo "    --clips ~/Movies/MyTrip \\"
echo "    --song ~/Music/song.mp3 \\"
echo "    --output ~/Movies/aftermovie.mp4"

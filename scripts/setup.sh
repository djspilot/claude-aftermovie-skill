#!/usr/bin/env bash
# aftermovie/scripts/setup.sh — one-time setup, idempotent.
# Installs the `aftermovie` package into a local venv so the user's global
# Python stays clean.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${HOME}/.skills-data/aftermovie"
VENV="${DATA_DIR}/venv"

echo "==> aftermovie setup"
echo "    Skill:    ${SKILL_DIR}"
echo "    Data:     ${DATA_DIR}"

# ---- ffmpeg ------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ""
  echo "  ! ffmpeg not found. Install it first:"
  echo "      brew install ffmpeg          (macOS)"
  echo "      apt-get install -y ffmpeg    (Debian/Ubuntu)"
  echo ""
  exit 1
fi
echo "    ffmpeg:   $(ffmpeg -version | head -n1)"

# ---- Python (3.10 - 3.12) ----------------------------------------------------
pick_python() {
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      local v
      v="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      case "$v" in
        3.10|3.11|3.12) echo "$cand"; return 0 ;;
      esac
    fi
  done
  return 1
}

PY="$(pick_python || true)"
if [ -z "${PY}" ]; then
  echo "  ! Need Python 3.10-3.12. Install with: brew install python@3.12"
  exit 1
fi
echo "    python:   $("${PY}" --version) ($(command -v "${PY}"))"

# ---- venv --------------------------------------------------------------------
mkdir -p "${DATA_DIR}"
if [ ! -d "${VENV}" ]; then
  echo "    Creating venv at ${VENV}..."
  "${PY}" -m venv "${VENV}"
fi
"${VENV}/bin/pip" install --upgrade pip --quiet

# ---- install package ---------------------------------------------------------
echo "    Installing aftermovie..."
"${VENV}/bin/pip" install -e "${SKILL_DIR}" --quiet

# ---- self-check --------------------------------------------------------------
echo ""
"${VENV}/bin/aftermovie" doctor

# ---- record paths ------------------------------------------------------------
cat > "${DATA_DIR}/config.json" <<EOF
{
  "venv": "${VENV}",
  "skill_dir": "${SKILL_DIR}",
  "data_dir": "${DATA_DIR}"
}
EOF

echo ""
echo "Setup complete."
echo ""
echo "Try it:"
echo "  ${VENV}/bin/aftermovie auto \\"
echo "    --clips ~/Movies/MyTrip \\"
echo "    --song ~/Music/song.mp3 \\"
echo "    --output ~/Movies/aftermovie.mp4"

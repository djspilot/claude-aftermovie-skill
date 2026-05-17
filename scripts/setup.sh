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

# ---- exiftool (optional — for single-file Live Photo extraction) -------------
if command -v exiftool >/dev/null 2>&1; then
  echo "    exiftool: $(exiftool -ver) (enables single-file Live Photo motion)"
else
  echo "    exiftool: not installed — install to extract motion from single-file"
  echo "              iPhone Live Photos / Pixel Motion Photos:"
  echo "                brew install exiftool          (macOS)"
  echo "                apt-get install -y libimage-exiftool-perl  (Debian/Ubuntu)"
fi

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
"${VENV}/bin/pip" install -e "${SKILL_DIR}[all]" --quiet 2>&1 | tail -3 || {
  echo "    ! mediapipe install failed — face detection / 9:16 reframing will be disabled."
  echo "    Falling back to install without [faces] extras..."
  "${VENV}/bin/pip" install -e "${SKILL_DIR}[mcp]" --quiet
}

# ---- MCP registration --------------------------------------------------------
# Write a stanza into ~/.claude/.mcp.json so Claude Code auto-spawns the server.
MCP_CONFIG="${HOME}/.claude/.mcp.json"
mkdir -p "$(dirname "${MCP_CONFIG}")"

"${VENV}/bin/python" - <<PYEOF
import json, os
path = "${MCP_CONFIG}"
venv_bin = "${VENV}/bin/aftermovie-mcp"
try:
    cfg = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}
cfg.setdefault("mcpServers", {})["aftermovie"] = {
    "command": venv_bin,
    "args": [],
}
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(cfg, open(path, "w"), indent=2)
print(f"    MCP:      registered 'aftermovie' in {path}")
PYEOF

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
echo "Try it from Claude Code:"
echo "  'make me an aftermovie from ~/Movies/MyTrip using ~/Music/song.mp3'"
echo ""
echo "Or from the shell:"
echo "  ${VENV}/bin/aftermovie auto \\"
echo "    --clips ~/Movies/MyTrip \\"
echo "    --song ~/Music/song.mp3 \\"
echo "    --output ~/Movies/aftermovie.mp4"
echo ""
echo "To uninstall the MCP entry, edit ${MCP_CONFIG} and remove the 'aftermovie' key."

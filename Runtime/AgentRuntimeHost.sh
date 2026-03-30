#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_SUPPORT="${MAC_ASSISTANT_APP_SUPPORT:-$HOME/Library/Application Support/MacAssistant}"
RUNTIME_HOME="$APP_SUPPORT/runtime"
VENV_DIR="$RUNTIME_HOME/venv"
STAMP_FILE="$RUNTIME_HOME/requirements.stamp"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
PYTHON_FILE="$SCRIPT_DIR/agent_runtime_host.py"

emit_progress() {
  local stage="$1"
  local message="$2"
  printf '{"type":"bootstrap_progress","stage":"%s","message":"%s"}\n' "$stage" "$message"
}

mkdir -p "$RUNTIME_HOME"
emit_progress "Checking local runtime" "Preparing the local Python runtime for voice and multimodal chat. This does not download the voice or agent models."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to bootstrap the local MLX runtime." >&2
  exit 1
fi

mkdir -p "$RUNTIME_HOME"
CURRENT_STAMP="$(shasum "$REQUIREMENTS_FILE" | awk '{print $1}')"

if [ ! -d "$VENV_DIR" ]; then
  emit_progress "Creating Python environment" "First launch creates an isolated Python 3.11 environment in Application Support."
  uv venv "$VENV_DIR" --python 3.11 >/dev/null 2>&1
else
  emit_progress "Checking Python environment" "Reusing the cached Python environment."
fi

if [ ! -f "$STAMP_FILE" ] || [ "$CURRENT_STAMP" != "$(cat "$STAMP_FILE")" ]; then
  emit_progress "Installing runtime packages" "Installing MLX, Hugging Face, PyTorch image-processor, and MCP bridge dependencies."
  source "$VENV_DIR/bin/activate"
  uv pip install --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS_FILE" >/dev/null 2>&1
  printf "%s" "$CURRENT_STAMP" > "$STAMP_FILE"
else
  emit_progress "Checking runtime packages" "Python voice and multimodal dependencies are already installed."
  source "$VENV_DIR/bin/activate"
fi

emit_progress "Starting local runtime host" "Launching the bundled runtime and checking installed model state."
exec "$VENV_DIR/bin/python" "$PYTHON_FILE"

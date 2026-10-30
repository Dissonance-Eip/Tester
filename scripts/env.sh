# shellcheck shell=bash
# Shared settings and helpers for setup.sh and cleanup.sh. Sourced, not run.
#
# Single source of truth for the shell scripts. Keep in sync with:
#   - PYTHON_VERSION  <-> the --python-version in the requirements.in header
#   - HF_MODEL_REPOS  <-> the Hub repos tester/config.py loads
#   - MAEST_CHECKPOINTS <-> the checkpoint MAEST_ARCH downloads via torch.hub

# These variables are consumed by the scripts that source this file.
# shellcheck disable=SC2034

PYTHON_VERSION="3.11.9"
VENV_DIR="venv"
# Records what setup.sh itself created, so cleanup.sh removes only that.
SETUP_STATE_FILE=".setup-state"

# Hugging Face Hub repos downloaded on first run (see tester/config.py).
HF_MODEL_REPOS=(
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo"  # WHISPER_SIZE + WHISPER_REVISION
    "m-a-p/MERT-v1-330M"                            # ENCODER_MODEL + MERT_REVISION
)

# MAEST weights come from a GitHub release via torch.hub, not from the Hub.
MAEST_CHECKPOINTS=(
    "discogs-maest-30s-pw-129e-swa.ckpt"  # MAEST_ARCH
)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# LOG_TAG is set by the sourcing script ("SETUP", "CLEAN").
log()   { printf '%b[%s]%b %s\n' "$GREEN" "${LOG_TAG:-INFO}" "$NC" "$1"; }
warn()  { printf '%b[WARN]%b  %s\n' "$YELLOW" "$NC" "$1"; }
error() { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1" >&2; exit 1; }

# confirm "question" -> 0 on y/Y, 1 otherwise. Default is No.
# ASSUME_YES=true answers yes; with no terminal on stdin the answer is No.
confirm() {
    local answer=""
    if [[ "${ASSUME_YES:-false}" == true ]]; then
        log "$1 [y/N] y (--yes)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        log "$1 [y/N] n (no terminal)"
        return 1
    fi
    read -r -p "$(printf '%b[?]%b  %s [y/N] ' "$YELLOW" "$NC" "$1")" answer || answer=""
    [[ "$answer" =~ ^[Yy]$ ]]
}

# Hub cache directory, resolved the way huggingface_hub resolves it.
hf_hub_cache() {
    if [[ -n "${HF_HUB_CACHE:-}" ]]; then
        printf '%s\n' "$HF_HUB_CACHE"
    elif [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
        printf '%s\n' "$HUGGINGFACE_HUB_CACHE"
    elif [[ -n "${HF_HOME:-}" ]]; then
        printf '%s\n' "$HF_HOME/hub"
    else
        printf '%s\n' "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub"
    fi
}

# torch.hub checkpoint directory, resolved the way torch resolves it.
torch_hub_checkpoints() {
    local torch_home="${TORCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/torch}"
    printf '%s\n' "$torch_home/hub/checkpoints"
}

# "org/name" -> "models--org--name" (the Hub cache folder name).
hf_repo_dir() {
    printf 'models--%s\n' "${1//\//--}"
}

# Directory of the pyenv-managed interpreter for PYTHON_VERSION.
pyenv_python_prefix() {
    printf '%s\n' "${PYENV_ROOT:-$HOME/.pyenv}/versions/$PYTHON_VERSION"
}

# Python version reported by an interpreter, or empty if it can't run.
python_version_of() {
    "$1" -c 'import platform; print(platform.python_version())' 2>/dev/null || true
}

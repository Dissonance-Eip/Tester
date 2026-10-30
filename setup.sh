#!/bin/bash
# ============================================================
# Music Perturbation Tester - Setup Script
# Installs everything from scratch. Safe to re-run: steps that are
# already done are skipped. Never edits shell rc files without asking.
#
# Usage: ./setup.sh [--yes]
#   --yes   answer yes to every prompt (rebuilding a venv on the wrong
#           Python, adding Homebrew to your login shell file)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck source-path=SCRIPTDIR source=scripts/env.sh
source scripts/env.sh
LOG_TAG="SETUP"
ASSUME_YES=false

BREW_PKGS=(ffmpeg pkg-config pyenv openssl readline sqlite3 xz zlib)

usage() {
    echo "Usage: ./setup.sh [--yes]"
    echo "  --yes   answer yes to every prompt"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y)  ASSUME_YES=true ;;
            --help|-h) usage; exit 0 ;;
            *)         usage; error "Unknown option: $1" ;;
        esac
        shift
    done
}

# record_state KEY VALUE: note something setup created. Write-once per line,
# so a later run never overwrites what an earlier run recorded.
record_state() {
    local line="$1=$2"
    if ! grep -qxF "$line" "$SETUP_STATE_FILE" 2>/dev/null; then
        printf '%s\n' "$line" >> "$SETUP_STATE_FILE"
    fi
}

# ── 0. System check ──────────────────────────────────────────
check_system() {
    log "Checking system..."
    if [[ "$(uname)" != "Darwin" ]]; then error "macOS only."; fi
    if [[ "$(uname -m)" != "arm64" ]]; then error "Apple Silicon (M1/M2/M3/M4) required."; fi
    log "macOS Apple Silicon."
}

# ── 1. Xcode Command Line Tools ──────────────────────────────
ensure_xcode_clt() {
    log "Checking Xcode Command Line Tools..."
    if ! xcode-select -p >/dev/null 2>&1; then
        warn "Xcode Command Line Tools not found."
        warn "A dialog will appear: click 'Install', wait for it to finish, then press Enter here."
        xcode-select --install 2>/dev/null || true
        read -r -p "  Press Enter once the Xcode CLT installation has completed... " || true
        xcode-select -p >/dev/null 2>&1 || error "Xcode Command Line Tools still not found. Re-run after installing."
    fi
    log "Xcode Command Line Tools ready."
}

# ── 2. Homebrew ───────────────────────────────────────────────
login_shell_file() {
    case "$(basename "${SHELL:-}")" in
        zsh)  printf '%s\n' "$HOME/.zprofile" ;;
        bash) printf '%s\n' "$HOME/.bash_profile" ;;
        *)    printf '\n' ;;
    esac
}

# Homebrew wasn't on PATH: offer (default No) to add it to the login shell file.
offer_brew_on_path() {
    local rc_file line
    rc_file="$(login_shell_file)"
    # Single quotes on purpose: the line is written literally, expanded at login.
    # shellcheck disable=SC2016
    line='eval "$(/opt/homebrew/bin/brew shellenv)"'
    if [[ -n "$rc_file" ]] && grep -qF "brew shellenv" "$rc_file" 2>/dev/null; then
        return 0
    fi
    if [[ -n "$rc_file" ]] && confirm "Add Homebrew to your PATH in $rc_file?"; then
        printf '\n# Homebrew\n%s\n' "$line" >> "$rc_file"
        log "Added Homebrew to $rc_file."
    else
        warn "Homebrew is not on your PATH in new shells. To add it, put this in your login shell file:"
        warn "  $line"
    fi
}

ensure_homebrew() {
    log "Checking Homebrew..."
    if ! command -v brew >/dev/null 2>&1 && [[ ! -x /opt/homebrew/bin/brew ]]; then
        warn "Homebrew not found. Installing..."
        NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
            || error "Homebrew install failed."
    fi
    if ! command -v brew >/dev/null 2>&1; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        offer_brew_on_path
    fi
    log "Homebrew ready."
}

# ── 3. Core system packages ───────────────────────────────────
ensure_brew_packages() {
    local pkg
    log "Checking system packages..."
    for pkg in "${BREW_PKGS[@]}"; do
        if ! brew list --formula "$pkg" >/dev/null 2>&1; then
            log "  Installing $pkg..."
            brew install "$pkg" || error "Failed to install $pkg."
            record_state brew_package "$pkg"
        fi
    done
    log "System packages ready."
}

# Point compilers at Homebrew's libraries, once, for both the pyenv Python
# build (ssl, sqlite, readline, lzma) and any C extension pip builds.
export_brew_build_flags() {
    local prefix lib ldflags="" cppflags="" pkgconfig=""
    prefix="$(brew --prefix)"
    for lib in openssl readline sqlite3 xz zlib; do
        ldflags+=" -L$prefix/opt/$lib/lib"
        cppflags+=" -I$prefix/opt/$lib/include"
        pkgconfig+="$prefix/opt/$lib/lib/pkgconfig:"
    done
    export LDFLAGS="${ldflags# } -L$prefix/lib"
    export CPPFLAGS="${cppflags# } -I$prefix/include"
    export PKG_CONFIG_PATH="${pkgconfig}$prefix/lib/pkgconfig:$prefix/share/pkgconfig"
}

# ── 4. pyenv + pinned Python ─────────────────────────────────
ensure_python() {
    local prefix
    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    prefix="$(pyenv_python_prefix)"

    log "Checking Python $PYTHON_VERSION..."
    if [[ ! -x "$prefix/bin/python3" ]]; then
        warn "Installing Python $PYTHON_VERSION (compiling from source, a few minutes)..."
        pyenv install --skip-existing "$PYTHON_VERSION" || error "Python $PYTHON_VERSION build failed."
        record_state pyenv_python "$PYTHON_VERSION"
    fi

    if [[ "$(cat .python-version 2>/dev/null || true)" != "$PYTHON_VERSION" ]]; then
        if [[ ! -e .python-version ]]; then record_state python_version_file .python-version; fi
        pyenv local "$PYTHON_VERSION"
    fi

    PYTHON="$prefix/bin/python3"
    local found
    found="$(python_version_of "$PYTHON")"
    if [[ "$found" != "$PYTHON_VERSION" ]]; then
        error "Expected Python $PYTHON_VERSION at $PYTHON, found '${found:-nothing}'."
    fi
    log "Python $PYTHON_VERSION at $PYTHON."
}

# ── 5. Virtual environment ───────────────────────────────────
ensure_venv() {
    local found
    log "Checking virtual environment..."
    if [[ -e "$VENV_DIR" && ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
        error "$VENV_DIR/ exists but is not a virtualenv. Move it away and re-run."
    fi
    if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
        found="$(python_version_of "$VENV_DIR/bin/python")"
        if [[ "$found" == "$PYTHON_VERSION" ]]; then
            log "Reusing $VENV_DIR/ (Python $found)."
            return 0
        fi
        warn "$VENV_DIR/ uses Python '${found:-unknown}', expected $PYTHON_VERSION."
        confirm "Delete $VENV_DIR/ and rebuild it?" || error "Keeping the old $VENV_DIR/. Remove it yourself and re-run."
        rm -rf -- "$VENV_DIR"
    fi
    "$PYTHON" -m venv "$VENV_DIR"
    record_state venv "$VENV_DIR"
    log "Created $VENV_DIR/."
}

# ── 6. Python dependencies ───────────────────────────────────
install_dependencies() {
    local venv_python="$VENV_DIR/bin/python"
    log "Installing pinned dependencies from requirements.txt (torch included)..."
    "$venv_python" -m pip install --upgrade pip --quiet
    "$venv_python" -m pip install --prefer-binary -r requirements.txt \
        || error "Failed to install from requirements.txt."
    log "Dependencies ready."
}

# ── 7. MPS fallback (in the venv, not your shell rc) ─────────
enable_mps_fallback() {
    local activate="$VENV_DIR/bin/activate"
    if ! grep -qF "PYTORCH_ENABLE_MPS_FALLBACK" "$activate"; then
        printf '\n# PyTorch MPS fallback for Apple Silicon (added by setup.sh)\nexport PYTORCH_ENABLE_MPS_FALLBACK=1\n' >> "$activate"
    fi
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    log "PYTORCH_ENABLE_MPS_FALLBACK=1 is set by $activate."
}

# ── 8. Sanity check ──────────────────────────────────────────
sanity_check() {
    log "Running sanity check..."
    "$VENV_DIR/bin/python" test.py || error "Sanity check failed, see above for details."
}

main() {
    parse_args "$@"
    check_system
    ensure_xcode_clt
    ensure_homebrew
    ensure_brew_packages
    export_brew_build_flags
    ensure_python
    ensure_venv
    install_dependencies
    enable_mps_fallback
    sanity_check

    printf '\n%b============================================%b\n' "$GREEN" "$NC"
    printf '%b  All done! Ready to test perturbations.   %b\n' "$GREEN" "$NC"
    printf '%b============================================%b\n\n' "$GREEN" "$NC"
    echo "  Activate env:  source $VENV_DIR/bin/activate"
    echo "  Run tester:    python3 run.py original.wav perturbed.wav"
    echo ""
}

main "$@"

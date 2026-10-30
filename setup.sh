#!/bin/bash

# ============================================================
# Music Perturbation Tester - Setup Script
# No prerequisites required — installs everything from scratch.
# ============================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 0. System check ──────────────────────────────────────────
log "Checking system..."
[[ "$(uname)"    != "Darwin" ]] && error "macOS only."
[[ "$(uname -m)" != "arm64"  ]] && error "Apple Silicon (M1/M2/M3) required."
log "macOS Apple Silicon."

# ── 1. Xcode Command Line Tools ──────────────────────────────
log "Checking Xcode Command Line Tools..."
if ! xcode-select -p &>/dev/null; then
    warn "Xcode Command Line Tools not found."
    warn "A dialog will appear — click 'Install' and wait for it to finish, then press Enter here."
    xcode-select --install 2>/dev/null || true
    read -r -p "  Press Enter once the Xcode CLT installation has completed... "
    xcode-select -p &>/dev/null || error "Xcode Command Line Tools still not found. Re-run after installing."
fi
log "Xcode Command Line Tools ready."

# ── 2. Homebrew ───────────────────────────────────────────────
log "Checking Homebrew..."
if ! command -v brew &>/dev/null && [[ ! -x /opt/homebrew/bin/brew ]]; then
    warn "Homebrew not found. Installing..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
        || error "Homebrew install failed."
fi
# Ensure Homebrew is in PATH for this session and future shells
if ! command -v brew &>/dev/null; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
if ! grep -q "brew shellenv" ~/.zprofile 2>/dev/null; then
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
fi
log "Homebrew ready."

# ── 3. Core system packages ───────────────────────────────────
log "Installing system packages..."
BREW_PKGS=(ffmpeg pkg-config pyenv openssl readline sqlite3 xz zlib)
for pkg in "${BREW_PKGS[@]}"; do
    if ! brew list --formula "$pkg" &>/dev/null; then
        log "  Installing $pkg..."
        brew install "$pkg" || error "Failed to install $pkg."
    fi
done
log "System packages ready."

# ── 4. pyenv + Python 3.11.9 ─────────────────────────────────
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

if ! grep -q "pyenv init" ~/.zshrc 2>/dev/null; then
    {
        echo ''
        echo '# pyenv'
        echo 'export PYENV_ROOT="$HOME/.pyenv"'
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
        echo 'eval "$(pyenv init -)"'
    } >> ~/.zshrc
fi

REQUIRED_PYTHON="3.11.9"
log "Checking Python $REQUIRED_PYTHON..."
if ! pyenv versions 2>/dev/null | grep -q "$REQUIRED_PYTHON"; then
    warn "Installing Python $REQUIRED_PYTHON (compiling from source — a few minutes)..."

    # Point the compiler at Homebrew's OpenSSL/readline/etc. so Python
    # gets a full-featured build (ssl, sqlite, readline, lzma all work).
    BREW_PREFIX="$(brew --prefix)"
    export LDFLAGS="-L$BREW_PREFIX/opt/openssl/lib -L$BREW_PREFIX/opt/readline/lib -L$BREW_PREFIX/opt/sqlite3/lib -L$BREW_PREFIX/opt/xz/lib -L$BREW_PREFIX/opt/zlib/lib"
    export CPPFLAGS="-I$BREW_PREFIX/opt/openssl/include -I$BREW_PREFIX/opt/readline/include -I$BREW_PREFIX/opt/sqlite3/include -I$BREW_PREFIX/opt/xz/include -I$BREW_PREFIX/opt/zlib/include"
    export PKG_CONFIG_PATH="$BREW_PREFIX/opt/openssl/lib/pkgconfig:$BREW_PREFIX/opt/readline/lib/pkgconfig:$BREW_PREFIX/opt/sqlite3/lib/pkgconfig:$BREW_PREFIX/opt/xz/lib/pkgconfig:$BREW_PREFIX/opt/zlib/lib/pkgconfig:$BREW_PREFIX/lib/pkgconfig"

    pyenv install "$REQUIRED_PYTHON" || error "Python $REQUIRED_PYTHON build failed."
fi
pyenv local "$REQUIRED_PYTHON"
PYTHON="$(pyenv which python3)"
log "Python $REQUIRED_PYTHON at $PYTHON."

# ── 5. Virtual environment ───────────────────────────────────
log "Setting up virtual environment..."
if [[ -d "venv" ]]; then
    warn "venv exists — rebuilding with Python $REQUIRED_PYTHON..."
    rm -rf venv
fi
"$PYTHON" -m venv venv
source venv/bin/activate
log "venv ready."

# ── 6. Python dependencies ───────────────────────────────────
log "Installing Python dependencies..."
pip install --upgrade pip --quiet

# Expose Homebrew headers/libs for any C extensions that build from source
BREW_PREFIX="$(brew --prefix)"
export PKG_CONFIG_PATH="$BREW_PREFIX/lib/pkgconfig:$BREW_PREFIX/share/pkgconfig"
export LDFLAGS="-L$BREW_PREFIX/lib"
export CPPFLAGS="-I$BREW_PREFIX/include"

# requirements.txt is the full lock (torch included), compiled from requirements.in.
log "Installing pinned dependencies..."
pip install --prefer-binary -r requirements.txt || error "Failed to install from requirements.txt."

log "Dependencies ready."

# ── 7. MPS fallback ──────────────────────────────────────────
log "Setting PYTORCH_ENABLE_MPS_FALLBACK..."
if ! grep -q "PYTORCH_ENABLE_MPS_FALLBACK" ~/.zshrc 2>/dev/null; then
    {
        echo ''
        echo '# PyTorch MPS fallback for Apple Silicon'
        echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1'
    } >> ~/.zshrc
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
log "MPS fallback ready."

# ── 8. Sanity check ──────────────────────────────────────────
log "Running sanity check..."
python3 test.py || error "Sanity check failed — see above for details."

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  All done! Ready to test perturbations.   ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Activate env:  source venv/bin/activate"
echo "  Run tester:    python3 run.py original.wav perturbed.wav"
echo ""

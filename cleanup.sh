#!/bin/bash

# ============================================================
# Music Perturbation Tester - Cleanup Script
# Removes the venv, downloaded models, and optionally the
# pyenv Python install created by setup.sh.
# Does NOT touch Homebrew packages (used system-wide).
# ============================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[CLEAN]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
confirm() {
    read -r -p "$(echo -e "${YELLOW}[?]${NC}  $1 [y/N] ")" ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

echo ""
echo -e "${RED}============================================${NC}"
echo -e "${RED}  Music Perturbation Tester — Cleanup      ${NC}"
echo -e "${RED}============================================${NC}"
echo ""

# ── 1. Virtual environment ───────────────────────────────────
if [[ -d "venv" ]]; then
    log "Removing venv..."
    rm -rf venv
    log "venv removed ✅"
else
    log "No venv found — skipping."
fi

# Remove pyenv local version pin
if [[ -f ".python-version" ]]; then
    rm -f .python-version
    log ".python-version removed ✅"
fi

# ── 2. HuggingFace model cache ───────────────────────────────
HF_CACHE="$HOME/.cache/huggingface/hub"
MODELS=(
    "models--Systran--faster-whisper-base"
    "models--Systran--faster-whisper-large-v3"
    "models--Systran--faster-whisper-large-v3-turbo"
    "models--m-a-p--MERT-v1-330M"
    "models--mtg-upf--discogs-maest-30s-pw-129e"
    "models--MIT--ast-finetuned-audioset-10-10-0.4593"
    # legacy — only present if you ran an older version of the tester
    "models--facebook--musicgen-melody"
    "models--facebook--musicgen-small"
    "models--facebook--musicgen-large"
    "models--laion--larger_clap_music"
    "models--dima806--music_genres_classification"
)

echo ""
log "Checking HuggingFace model cache..."
found_any=false
for model in "${MODELS[@]}"; do
    model_path="$HF_CACHE/$model"
    if [[ -d "$model_path" ]]; then
        size=$(du -sh "$model_path" 2>/dev/null | cut -f1)
        echo "    $model  ($size)"
        found_any=true
    fi
done

if $found_any; then
    echo ""
    if confirm "Remove the above models from HuggingFace cache?"; then
        for model in "${MODELS[@]}"; do
            model_path="$HF_CACHE/$model"
            if [[ -d "$model_path" ]]; then
                rm -rf "$model_path"
                log "$model removed ✅"
            fi
        done
    else
        log "Model cache kept."
    fi
else
    log "No cached models found — skipping."
fi

# ── 3. pyenv Python 3.11.9 ───────────────────────────────────
PYENV_PYTHON="$HOME/.pyenv/versions/3.11.9"
echo ""
if [[ -d "$PYENV_PYTHON" ]]; then
    size=$(du -sh "$PYENV_PYTHON" 2>/dev/null | cut -f1)
    if confirm "Remove pyenv Python 3.11.9 ($size)? Only do this if nothing else uses it."; then
        rm -rf "$PYENV_PYTHON"
        log "Python 3.11.9 removed ✅"
    else
        log "Python 3.11.9 kept."
    fi
else
    log "No pyenv Python 3.11.9 found — skipping."
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
log "Cleanup complete. Run ./setup.sh to reinstall."
echo ""

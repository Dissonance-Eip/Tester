#!/bin/bash
# ============================================================
# Music Perturbation Tester - Cleanup Script
# Removes what setup.sh and the tester created, and nothing else:
#   - ./venv and the ./.python-version pin
#   - this tester's models in the Hugging Face and torch.hub caches
#     (honours HF_HUB_CACHE / HF_HOME / TORCH_HOME / XDG_CACHE_HOME)
#   - the pyenv Python, only if setup.sh installed it
# Every removal is confirmed (default No). Homebrew packages are left alone.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck source-path=SCRIPTDIR source=scripts/env.sh
source scripts/env.sh
LOG_TAG="CLEAN"
ASSUME_YES=false
DRY_RUN=false
REMOVED=()

usage() {
    echo "Usage: ./cleanup.sh [--dry-run] [--yes]"
    echo "  --dry-run   list what would be removed, remove nothing"
    echo "  --yes       remove without asking"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run|-n) DRY_RUN=true ;;
            --yes|-y)     ASSUME_YES=true ;;
            --help|-h)    usage; exit 0 ;;
            *)            usage; error "Unknown option: $1" ;;
        esac
        shift
    done
}

size_of() { du -sh "$1" 2>/dev/null | cut -f1; }

# remove_paths "question" PATH...: confirm once, then delete each path.
# In dry-run mode only reports what would go.
remove_paths() {
    local question="$1" path
    shift
    if $DRY_RUN; then
        for path in "$@"; do log "[dry-run] would remove $path"; done
        return 0
    fi
    confirm "$question" || { log "Kept."; return 1; }
    for path in "$@"; do
        rm -rf -- "$path"
        REMOVED+=("$path")
        log "Removed $path"
    done
}

# ── 1. Project environment ───────────────────────────────────
clean_project_env() {
    local targets=()
    if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
        targets+=("$PWD/$VENV_DIR")
        log "Found $PWD/$VENV_DIR ($(size_of "$VENV_DIR"))"
    elif [[ -e "$VENV_DIR" ]]; then
        warn "$PWD/$VENV_DIR is not a virtualenv, leaving it alone."
    fi
    if [[ "$(cat .python-version 2>/dev/null || true)" == "$PYTHON_VERSION" ]]; then
        targets+=("$PWD/.python-version")
        log "Found $PWD/.python-version ($PYTHON_VERSION)"
    fi
    if [[ ${#targets[@]} -eq 0 ]]; then
        log "No venv or .python-version pin found, skipping."
        return 0
    fi
    if remove_paths "Remove the virtualenv and Python pin above?" "${targets[@]}" && ! $DRY_RUN; then
        forget_state venv
        forget_state python_version_file
    fi
}

# ── 2. Model caches ──────────────────────────────────────────
clean_models() {
    local hub ckpt_dir repo dir name targets=()
    hub="$(hf_hub_cache)"
    ckpt_dir="$(torch_hub_checkpoints)"
    log "Hugging Face cache: $hub"
    log "torch.hub cache:    $ckpt_dir"
    for repo in "${HF_MODEL_REPOS[@]}"; do
        dir="$(hf_repo_dir "$repo")"
        if [[ -d "$hub/$dir" ]]; then
            targets+=("$hub/$dir")
            log "Found $repo ($(size_of "$hub/$dir"))"
        fi
        if [[ -d "$hub/.locks/$dir" ]]; then targets+=("$hub/.locks/$dir"); fi
    done
    for name in "${MAEST_CHECKPOINTS[@]}"; do
        if [[ -f "$ckpt_dir/$name" ]]; then
            targets+=("$ckpt_dir/$name")
            log "Found $name ($(size_of "$ckpt_dir/$name"))"
        fi
    done
    if [[ ${#targets[@]} -eq 0 ]]; then
        log "No cached tester models found, skipping."
        return 0
    fi
    remove_paths "Remove these models? Other projects sharing the cache will re-download them." \
        "${targets[@]}" || true
}

# ── 3. pyenv Python (only if setup.sh installed it) ──────────
clean_python() {
    local prefix
    prefix="$(pyenv_python_prefix)"
    if ! grep -qxF "pyenv_python=$PYTHON_VERSION" "$SETUP_STATE_FILE" 2>/dev/null; then
        log "pyenv Python $PYTHON_VERSION was not installed by setup.sh, leaving it alone."
        return 0
    fi
    if [[ ! -d "$prefix" ]]; then
        forget_state pyenv_python
        return 0
    fi
    log "Found pyenv Python $PYTHON_VERSION, installed by setup.sh ($(size_of "$prefix"))"
    if $DRY_RUN; then
        log "[dry-run] would run: pyenv uninstall -f $PYTHON_VERSION"
        return 0
    fi
    if confirm "Uninstall pyenv Python $PYTHON_VERSION? Only if nothing else uses it."; then
        pyenv uninstall -f "$PYTHON_VERSION"
        REMOVED+=("$prefix")
        forget_state pyenv_python
        log "Uninstalled Python $PYTHON_VERSION."
    else
        log "Kept."
    fi
}

# forget_state KEY: drop KEY's lines from the state file; delete it once empty.
forget_state() {
    local remaining
    [[ -f "$SETUP_STATE_FILE" ]] || return 0
    remaining="$(grep -v "^$1=" "$SETUP_STATE_FILE" || true)"
    if [[ -z "$remaining" ]]; then
        rm -f -- "$SETUP_STATE_FILE"
    else
        printf '%s\n' "$remaining" > "$SETUP_STATE_FILE"
    fi
}

report_brew_packages() {
    local pkgs
    [[ -f "$SETUP_STATE_FILE" ]] || return 0
    pkgs="$(sed -n 's/^brew_package=//p' "$SETUP_STATE_FILE" | tr '\n' ' ')"
    if [[ -n "$pkgs" ]]; then
        log "setup.sh installed these Homebrew packages, left in place: $pkgs"
    fi
}

summary() {
    local path
    echo ""
    if $DRY_RUN; then
        log "Dry run: nothing was removed."
    elif [[ ${#REMOVED[@]} -eq 0 ]]; then
        log "Nothing was removed."
    else
        log "Removed:"
        for path in "${REMOVED[@]}"; do echo "    $path"; done
    fi
    log "Run ./setup.sh to reinstall."
}

main() {
    parse_args "$@"
    printf '\n%b  Music Perturbation Tester: Cleanup%b\n\n' "$RED" "$NC"
    clean_project_env
    echo ""
    clean_models
    echo ""
    clean_python
    report_brew_packages
    summary
}

main "$@"

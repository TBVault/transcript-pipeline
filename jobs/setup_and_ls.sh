#!/bin/bash
# Job: Clone transcript-pipeline on igpu15 (if not present) and ls the structure

set -e

REPO_URL="https://github.com/TBVault/transcript-pipeline"
REPO_DIR="$HOME/transcript-pipeline"

echo "=== Hostname ==="
hostname

echo "=== Conda env ==="
conda info --envs | grep vdabase || echo "vdabase not listed"

echo "=== Checking repo ==="
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists at $REPO_DIR"
else
    echo "Cloning repo to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "=== Repo structure (ls) ==="
ls "$REPO_DIR"

echo "=== Done ==="

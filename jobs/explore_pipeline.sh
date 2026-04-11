#!/bin/bash
# Job: Explore transcript-pipeline structure and /dev/shm/organized_mp3 on igpu15

set -e

REPO_URL="https://github.com/TBVault/transcript-pipeline"
REPO_DIR="$HOME/transcript-pipeline"

echo "=== 1. Hostname ==="
hostname

echo "=== 2. Conda env ==="
conda info --envs | grep '*' || true

echo "=== 3. Clone repo (if not present) ==="
cd ~ && git clone "$REPO_URL" || echo "repo already exists"

echo "=== 4. ls -la $REPO_DIR ==="
ls -la "$REPO_DIR"

echo "=== 5. Pipeline subdirectory listings ==="
for d in 01_preprocessing 02_transcription 03_timestamp_alignment 04_diarization 05_speaker_identity 06_postprocessing 07_evaluation 08_utilities orchestration; do
    echo "--- $d ---"
    ls "$REPO_DIR/$d" 2>/dev/null || echo "(not found)"
done

echo "=== 6. /dev/shm/organized_mp3 (first 20) ==="
ls /dev/shm/organized_mp3/ 2>/dev/null | head -20 || echo "(directory not found)"

echo "=== Done ==="

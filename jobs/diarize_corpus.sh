#!/usr/bin/env bash
# Stage A of the speaker-identity rollout: pyannote diarization over the whole
# tbv_mp3 corpus. Multi-GPU, idempotent (skips any stem that already has a
# <stem>.txt). Writes pyannote turns to outputs/05_pyannote_diarization/.
# Coexists with the running tbv_batch_01.sh — V100s are 32GB, pyannote needs
# only a few GB on top of whisper.
set -uo pipefail

REPO="/lab/kiran/transcript-pipeline"
PYTHON="/home3/kiran/anaconda3/envs/vdabase/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

export AUDIO_ROOT="/lab/kiran/tbv_mp3"
export PYANNOTE_OUTPUT_DIR="$REPO/outputs/05_pyannote_diarization"
# HF_TOKEN unset -> the script falls back to the cached HF login.
mkdir -p "$PYANNOTE_OUTPUT_DIR"

echo "[diarize_corpus] start: $(ls "$AUDIO_ROOT"/*.mp3 | wc -l) mp3s, "\
"$(ls "$PYANNOTE_OUTPUT_DIR"/*.txt 2>/dev/null | wc -l) already done"
"$PYTHON" "$REPO/04_diarization/diarization_from_segments_local.py"
echo "[diarize_corpus] done: $(ls "$PYANNOTE_OUTPUT_DIR"/*.txt 2>/dev/null | wc -l) txt files"

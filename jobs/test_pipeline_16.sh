#!/usr/bin/env bash
# test_pipeline_16.sh — torch.load monkey-patched to weights_only=False
set -euo pipefail
CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"
REPO_DIR="/lab/kiran/transcript-pipeline"
echo "=== test_pipeline_16 on $(hostname) at $(date) ==="
cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true
# Clear previous whisper outputs so we actually run transcription
rm -rf outputs/02_whisper_transcripts/
mkdir -p outputs/02_whisper_transcripts
export CUDA_VISIBLE_DEVICES="0"
# Pick one file and run whisper transcription
FNAME="2016-18_part-1_2018：_A_Retrospective-10156641486771265"
MP3="/dev/shm/organized_mp3/$FNAME/$FNAME.mp3"
OUT_DIR="outputs/02_whisper_transcripts/$FNAME"
echo "Transcribing $FNAME..."
$PYTHON 02_transcription/whisper_transcribe.py "$MP3" --output_dir "$OUT_DIR"
echo "=== DONE ==="

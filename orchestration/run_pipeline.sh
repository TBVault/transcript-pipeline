#!/usr/bin/env bash
# run_pipeline.sh - End-to-End Single-Lecture Pipeline
# Usage: ./run_pipeline.sh <lecture_id>
set -euo pipefail
LID="${1:?Usage: ./run_pipeline.sh <lecture_id>}"
echo "=== Pipeline: ${LID} ==="
echo "[1/7] MP3 header check..."
python 01_preprocessing/check_mp3_headers.py "${AUDIO_ROOT}/" --fix 2>/dev/null || true
echo "[2/7] Kirtan detection..."
python 01_preprocessing/detect_kirtans.py "${LID}"
echo "[3/7] Gemini transcription..."
python 02_transcription/gemini_transcribe.py "${AUDIO_ROOT}/${LID}.mp3"
echo "[3/7] Whisper transcription..."
python 02_transcription/whisper_transcribe.py "${AUDIO_ROOT}/${LID}.mp3"
echo "[4/7] Timestamp alignment..."
python 03_timestamp_alignment/fuzz.py "${WHISPER_OUTPUT_DIR}/${LID}/"
python 03_timestamp_alignment/correct_gemini_timesteps.py "${WHISPER_OUTPUT_DIR}/${LID}/"
echo "[5/7] Diarization..."
python 04_diarization/split_audio_dia.py "${LID}"
echo "[6/7] Speaker embeddings..."
python 05_speaker_identity/gen_embeddings.py "${AUDIO_ROOT}/${LID}/"
echo "[7/7] Speaker merge..."
python 06_postprocessing/speaker_merge.py "${LID}"
echo "=== Done: ${LID} ==="

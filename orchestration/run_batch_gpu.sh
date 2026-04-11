#!/usr/bin/env bash
# run_batch_gpu.sh - Multi-GPU Batch Pipeline
# Usage: ./run_batch_gpu.sh [output_dir]
set -euo pipefail
OUT="${1:-v2}"
echo "=== Batch ($(nvidia-smi -L 2>/dev/null | wc -l || echo 1) GPUs) ==="
echo "[1/3] Multi-GPU diarization..."
python 04_diarization/diarization_from_segments_local.py
echo "[2/3] Global clustering..."
python 05_speaker_identity/global_clustering.py "${VOICEPRINTS_DIR}" global_map.json
echo "[3/3] Batch merge..."
python 06_postprocessing/batch_parallel.py --output_dir "${OUT}"
echo "=== Done -> ${OUT}/ ==="

#!/usr/bin/env bash
# diarize_fusion_batch.sh — Fusion diarization (pyannote + DiariZen via
# DOVER-Lap) over tbv_mp3. Existing pyannote turns are reused; only DiariZen
# runs fresh. Idempotent (skips stems with a fusion .txt).
#
# Env: FUSION_LIMIT=N   stop after N new files (0 = all)
#      GPUS="2 3"       which GPUs to use (default: 2 3, leaving 0-1 free)
# Output: outputs/06_fusion_diarization/<stem>.txt
# Requires the diarizen overlay venv — see docs/speaker_identity.md.
set -uo pipefail

REPO="/lab/kiran/transcript-pipeline"
AUDIO="/lab/kiran/tbv_mp3"
OUT="$REPO/outputs/06_fusion_diarization"
LOGDIR="$REPO/outputs/tbv_logs"
PYTHON="/lab/kiran/diarizen_venv/bin/python"
export LD_LIBRARY_PATH="/home3/kiran/anaconda3/envs/tbv/lib"
LIMIT="${FUSION_LIMIT:-0}"
GPUS=(${GPUS:-2 3})

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR"
echo "=== diarize_fusion_batch at $(date) | GPUs: ${GPUS[*]} | limit: $LIMIT ==="

LIST=$(mktemp /tmp/fusion_pending.XXXXXX)
for f in "$AUDIO"/*.mp3; do
    stem="$(basename "$f" .mp3)"
    [ -s "$OUT/$stem.txt" ] && continue
    [ -f "$OUT/$stem.failed" ] && continue
    echo "$stem"
done > "$LIST"
[ "$LIMIT" -gt 0 ] && { head -n "$LIMIT" "$LIST" > "$LIST.l"; mv "$LIST.l" "$LIST"; }
echo "pending: $(wc -l < "$LIST")"

worker() {
    local slot="$1" gpu="$2"
    local i=0
    while IFS= read -r -u3 stem; do
        i=$((i + 1))
        [ $(( (i - 1) % ${#GPUS[@]} )) -ne "$slot" ] && continue
        echo "[GPU$gpu] $stem"
        if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
                "$REPO/04_diarization/diarize_fusion.py" "$AUDIO/$stem.mp3" \
                < /dev/null >> "$LOGDIR/$stem.fusion.log" 2>&1; then
            echo "[FAIL] $stem"
            touch "$OUT/$stem.failed"
        fi
    done 3< "$LIST"
}

PIDS=()
for s in "${!GPUS[@]}"; do
    worker "$s" "${GPUS[$s]}" &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
rm -f "$LIST"
echo "=== done at $(date): $(ls "$OUT"/*.txt 2>/dev/null | wc -l) fused, $(ls "$OUT"/*.failed 2>/dev/null | wc -l) failed ==="

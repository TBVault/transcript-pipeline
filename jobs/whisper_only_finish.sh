#!/usr/bin/env bash
# whisper_only_finish.sh — Finish the remaining tbv corpus WITHOUT the Gemini
# API (user directive 2026-07-01: no more Google transcription, ever).
#
# For every mp3 with no final JSON and no fail marker (i.e. the ~1,000 files
# that have no cached Gemini transcript), the transcript comes from Whisper
# alone: whisper_transcribe -> strip to {start,end,text} -> place in
# 04_fuzz_merged/<stem>/transcript.json (the slot speaker_merge and Stage C
# read) -> speaker_merge with the existing pyannote diarization + WavLM
# global_map -> outputs/08_final_json/<stem>.json.
#
# Whisper transcripts already produced by earlier runs (files that were stuck
# on [WAIT][gemini]) are reused as-is. Text quality is Whisper-grade (no
# Gemini fusion) — a WHISPER_ONLY marker file is left in the fuzz dir so these
# are distinguishable later.
#
# After the pass completes, chains Stage B + C (jobs/run_speaker_identity.sh)
# so the global map and ALL final JSONs are refreshed corpus-wide.
#
# Idempotent; per-file fail markers in outputs/tbv_failed/ (delete to retry).
# Env: WORKERS_PER_GPU (default 2), TBV_LIMIT=N (test mode), NUM_GPUS (4).
set -uo pipefail

CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"
REPO="/lab/kiran/transcript-pipeline"
AUDIO="/lab/kiran/tbv_mp3"
GBAK="/lab/kiran/gemini_3.0_flash"
OUT="$REPO/outputs"
FAILDIR="$OUT/tbv_failed"
LOGDIR="$OUT/tbv_logs"
NUM_GPUS="${NUM_GPUS:-4}"
WPG="${WORKERS_PER_GPU:-2}"
NWORK=$((NUM_GPUS * WPG))
LIMIT="${TBV_LIMIT:-0}"

cd "$REPO"
mkdir -p "$FAILDIR" "$LOGDIR" v2

echo "=== whisper_only_finish on $(hostname) at $(date) ==="
echo "Workers: $NWORK ($WPG per GPU x $NUM_GPUS GPUs) | Limit: $LIMIT"

valid_json() {
    [ -s "$1" ] && $PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d' "$1" 2>/dev/null
}

process_stem() {
    local stem="$1" gpu="$2"
    local mp3="$AUDIO/$stem.mp3"
    local wjson="$OUT/02_whisper_transcripts/$stem/transcript.json"
    local fdir="$OUT/04_fuzz_merged/$stem"
    local ftrans="$fdir/transcript.json"
    local final="$OUT/08_final_json/$stem.json"

    [ -s "$final" ] && return 0

    # Defensive: anything with a valid Gemini cache should have been finished
    # by tbv_batch_01 already — flag it instead of silently whisper-onlying it.
    if valid_json "$GBAK/$stem/$stem.json"; then
        echo "[UNEXPECTED-CACHED] $stem — has Gemini cache but no final JSON; skipping"
        return 0
    fi

    if ! valid_json "$wjson"; then
        [ -f "$FAILDIR/$stem.whisper" ] && return 0
        echo "[GPU$gpu][WHISPER] $stem"
        if ! CUDA_VISIBLE_DEVICES="$gpu" $PYTHON 02_transcription/whisper_transcribe.py \
                "$mp3" --output_dir "$OUT/02_whisper_transcripts/$stem" \
                >> "$LOGDIR/$stem.whisper.log" 2>&1 \
           || ! valid_json "$wjson"; then
            echo "[FAIL][whisper] $stem"
            touch "$FAILDIR/$stem.whisper"
            return 0
        fi
    fi

    if ! valid_json "$ftrans"; then
        mkdir -p "$fdir"
        $PYTHON - "$wjson" "$ftrans" <<'PYEOF' >> "$LOGDIR/$stem.align.log" 2>&1
import sys, json
with open(sys.argv[1]) as f:
    segs = json.load(f)
out = [{"start": s.get("start", 0.0), "end": s.get("end", 0.0),
        "text": s.get("text", "").strip()}
       for s in segs if s.get("text", "").strip()]
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"[WHISPER-ONLY] {len(out)} segments -> {sys.argv[2]}")
PYEOF
        if ! valid_json "$ftrans"; then
            echo "[FAIL][fuzz-empty] $stem"
            rm -f "$ftrans"
            touch "$FAILDIR/$stem.fuzz"
            return 0
        fi
        touch "$fdir/WHISPER_ONLY"
    fi

    [ -f "$FAILDIR/$stem.final" ] && return 0
    WHISPER_OUTPUT_DIR="$OUT/04_fuzz_merged" \
    PYANNOTE_OUTPUT_DIR="$OUT/05_pyannote_diarization" \
    GLOBAL_MAP_PATH="$OUT/07_speaker_clusters/global_map.json" \
    MAIN_SPEAKER_NAME="Vaisesika Dasa" \
    AUDIO_ROOT="$AUDIO" \
        $PYTHON 06_postprocessing/speaker_merge.py "$stem" >> "$LOGDIR/$stem.align.log" 2>&1
    [ -f "v2/$stem.json" ] && mv "v2/$stem.json" "$final"
    if valid_json "$final"; then
        echo "[DONE] $stem"
    else
        echo "[FAIL][final] $stem"
        rm -f "$final"
        touch "$FAILDIR/$stem.final"
    fi
}

worker() {
    local slot="$1" listfile="$2"
    local gpu=$((slot % NUM_GPUS))
    local i=0
    # fd 3: child processes (ffmpeg etc.) eat stdin and mangle the list
    while IFS= read -r -u3 stem; do
        i=$((i + 1))
        [ $(( (i - 1) % NWORK )) -ne "$slot" ] && continue
        process_stem "$stem" "$gpu" < /dev/null
    done 3< "$listfile"
}

PENDING_LIST=$(mktemp /tmp/whisper_only_pending.XXXXXX)
for f in "$AUDIO"/*.mp3; do
    [ -e "$f" ] || continue
    stem="$(basename "$f" .mp3)"
    [ -s "$OUT/08_final_json/$stem.json" ] && continue
    compgen -G "$FAILDIR/$stem.*" > /dev/null && continue
    echo "$stem"
done > "$PENDING_LIST"

if [ "$LIMIT" -gt 0 ]; then
    head -n "$LIMIT" "$PENDING_LIST" > "$PENDING_LIST.lim"
    mv "$PENDING_LIST.lim" "$PENDING_LIST"
fi

echo "Pending: $(wc -l < "$PENDING_LIST") files"

WORKER_PIDS=()
for slot in $(seq 0 $((NWORK - 1))); do
    worker "$slot" "$PENDING_LIST" &
    WORKER_PIDS+=($!)
done
for pid in "${WORKER_PIDS[@]}"; do wait "$pid"; done
rm -f "$PENDING_LIST"

echo "--- whisper pass done at $(date) ---"
echo "Final JSON: $(ls "$OUT/08_final_json" | wc -l) | Failures: $(ls "$FAILDIR" | wc -l)"

if [ "$LIMIT" -gt 0 ]; then
    echo "TBV_LIMIT set — skipping Stage B/C chain (test mode)"
    exit 0
fi

echo ""
echo "=== chaining Stage B + C (re-cluster + re-merge, corpus-wide) ==="
bash "$REPO/jobs/run_speaker_identity.sh" B C

echo ""
echo "=== whisper_only_finish COMPLETE at $(date) ==="
echo "Final JSON: $(ls "$OUT/08_final_json" | wc -l) | Failures: $(ls "$FAILDIR" | wc -l)"

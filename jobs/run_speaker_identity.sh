#!/usr/bin/env bash
# Full speaker-identity rollout over the tbv corpus, in three idempotent stages.
# Run after the diarization weights_only fix (04_diarization) and the
# diarization-aware clustering (05_speaker_identity/embed_and_cluster_diarized.py).
#
#   A  diarize       all mp3s          -> outputs/05_pyannote_diarization/<stem>.txt
#   B  embed+cluster all diarizations  -> outputs/07_speaker_clusters/global_map.json
#   C  speaker_merge all files w/ map  -> outputs/08_final_json/<stem>.json (re-merged)
#
# Usage:
#   jobs/run_speaker_identity.sh A B C     # any subset, in order; default "B C"
# Stage A is normally launched separately (jobs/diarize_corpus.sh) since it is
# the multi-hour long pole; pass A here to (re)run/resume it inline.
set -uo pipefail

REPO="/lab/kiran/transcript-pipeline"
AUDIO="/lab/kiran/tbv_mp3"
OUT="$REPO/outputs"
PYTHON="/home3/kiran/anaconda3/envs/vdabase/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

PYDIR="$OUT/05_pyannote_diarization"
GMAP="$OUT/07_speaker_clusters/global_map.json"
FUZZ="$OUT/04_fuzz_merged"
FINAL="$OUT/08_final_json"
LOGDIR="$OUT/tbv_logs"
mkdir -p "$PYDIR" "$OUT/07_speaker_clusters" "$FINAL"

STAGES="${*:-B C}"
valid_json() { [ -s "$1" ] && "$PYTHON" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1; }

run_A() {
    echo "=== Stage A: diarization ==="
    AUDIO_ROOT="$AUDIO" PYANNOTE_OUTPUT_DIR="$PYDIR" \
        "$PYTHON" "$REPO/04_diarization/diarization_from_segments_local.py"
    echo "  diarized: $(ls "$PYDIR"/*.txt 2>/dev/null | wc -l) files"
}

run_B() {
    echo "=== Stage B: embed + global cluster ==="
    MAIN_SPEAKER_NAME="Vaisesika Dasa" \
        "$PYTHON" "$REPO/05_speaker_identity/embed_and_cluster_diarized.py" \
        "$PYDIR" "$AUDIO" "$GMAP"
    valid_json "$GMAP" && echo "  global_map written: $GMAP" || { echo "  [FAIL] no global_map"; return 1; }
}

run_C() {
    echo "=== Stage C: re-merge final JSONs with speaker identities ==="
    local done=0 skip=0
    # Only files that already have a fuzz transcript can be (re)merged.
    for d in "$FUZZ"/*/; do
        [ -d "$d" ] || continue
        local stem; stem="$(basename "$d")"
        [ -s "$d/transcript.json" ] || { skip=$((skip+1)); continue; }
        WHISPER_OUTPUT_DIR="$FUZZ" \
        PYANNOTE_OUTPUT_DIR="$PYDIR" \
        GLOBAL_MAP_PATH="$GMAP" \
        MAIN_SPEAKER_NAME="Vaisesika Dasa" \
        AUDIO_ROOT="$AUDIO" \
            "$PYTHON" "$REPO/06_postprocessing/speaker_merge.py" "$stem" \
            >> "$LOGDIR/$stem.align.log" 2>&1 || { echo "  [FAIL][merge] $stem"; continue; }
        if [ -s "$REPO/v2/$stem.json" ]; then
            mv "$REPO/v2/$stem.json" "$FINAL/$stem.json"; done=$((done+1))
        fi
        [ $(( (done+skip) % 200 )) -eq 0 ] && echo "  ...re-merged $done, skipped $skip"
    done
    echo "  Stage C done: re-merged $done, skipped(no transcript) $skip"
}

for s in $STAGES; do
    case "$s" in
        A) run_A ;;
        B) run_B ;;
        C) run_C ;;
        *) echo "unknown stage: $s (use A B C)"; exit 1 ;;
    esac
done
echo "=== run_speaker_identity complete: stages [$STAGES] ==="

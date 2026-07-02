#!/usr/bin/env bash
# recover_gemini_failures.sh — re-run Gemini on files that died with the
# "CRITICAL ERROR: 'end'" bug (a single malformed segment killing the whole
# file). The KeyError is now fixed in stitch_segments, so these files can be
# transcribed successfully. On success we clear the .gemini fail marker so the
# running tbv_batch_01.sh picks the file up for whisper -> align -> final.
#
# Idempotent: skips files that already have a valid cache JSON.
set -uo pipefail

REPO="/lab/kiran/transcript-pipeline"
AUDIO="/lab/kiran/tbv_mp3"
GBAK="/lab/kiran/gemini_3.0_flash"
FAILDIR="$REPO/outputs/tbv_failed"
LOGDIR="$REPO/outputs/tbv_logs"
PYTHON="/home3/kiran/anaconda3/envs/vdabase/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

valid_json() { [ -s "$1" ] && "$PYTHON" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1; }

ok=0; fail=0; skipped=0
for marker in "$FAILDIR"/*.gemini; do
    [ -e "$marker" ] || continue
    stem="$(basename "$marker" .gemini)"
    bak="$GBAK/$stem/$stem.json"
    if valid_json "$bak"; then
        rm -f "$marker"; skipped=$((skipped+1))
        echo "[SKIP-HAVE-CACHE] $stem"; continue
    fi
    if [ ! -s "$AUDIO/$stem.mp3" ]; then
        echo "[NO-MP3] $stem"; fail=$((fail+1)); continue
    fi
    echo "[GEMINI-RETRY] $stem"
    if "$PYTHON" "$REPO/02_transcription/gemini_transcribe.py" "$AUDIO/$stem.mp3" \
            --output_dir "$GBAK/$stem" --threads 4 \
            < /dev/null >> "$LOGDIR/$stem.gemini.log" 2>&1 \
       && valid_json "$bak"; then
        rm -f "$marker"; ok=$((ok+1))
        echo "[GEMINI-RETRY-DONE] $stem"
    else
        fail=$((fail+1))
        echo "[GEMINI-RETRY-FAIL] $stem (marker kept)"
    fi
done

echo "=== recover_gemini_failures: recovered=$ok failed=$fail already-cached=$skipped ==="

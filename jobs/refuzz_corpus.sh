#!/usr/bin/env bash
# refuzz_corpus.sh — Re-run the full alignment chain (restored numba fuzz ->
# correct_gemini_timesteps -> greedypushing -> transcript build) over every
# Gemini-fused stem, then re-merge speakers and re-export.
#
# Why: the stripped fuzz that ran the corpus never emitted BRIDGE_SEGMENTs,
# so unaligned Gemini passages were lost (post-dedup) or duplicated
# (pre-dedup). The restored fuzz recovers them with interpolated timestamps.
# Whisper-only stems (WHISPER_ONLY marker) are skipped.
# Env: WORKERS (default 16)
set -uo pipefail

CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"
REPO="/lab/kiran/transcript-pipeline"
OUT="$REPO/outputs"
WORKERS="${WORKERS:-16}"
cd "$REPO"

echo "=== refuzz_corpus at $(date) | workers: $WORKERS ==="
export PYTHON REPO OUT

process_one() {
    local d="$1"
    local stem; stem="$(basename "$d")"
    [ -f "$d/WHISPER_ONLY" ] && return 0
    [ -s "$d/segments_with_whisper_and_gemini.json" ] || return 0
    if ! { $PYTHON 03_timestamp_alignment/fuzz.py "$d" \
           && $PYTHON 03_timestamp_alignment/correct_gemini_timesteps.py "$d" \
           && $PYTHON 03_timestamp_alignment/greedypushing_postprocess.py \
                  "$d/segments_corrected.json" "$d/segments_pushed.json"; \
         } >> "$OUT/tbv_logs/$stem.refuzz.log" 2>&1; then
        echo "[FAIL][refuzz] $stem"
        return 0
    fi
    $PYTHON - "$d/segments_pushed.json" "$d/transcript.json" <<'PYEOF' >> "$OUT/tbv_logs/$stem.refuzz.log" 2>&1
import sys, json
segs = json.load(open(sys.argv[1]))
out = []
for s in segs:
    if s.get("type") == "WHISPER_GROUP":
        continue
    t = (s.get("gemini_transcript") or s.get("whisper_transcript") or "").strip()
    if t:
        out.append({"start": s.get("start", 0.0), "end": s.get("end", 0.0), "text": t})
json.dump(out, open(sys.argv[2], "w"), indent=2, ensure_ascii=False)
print(f"[REFUZZ-BUILD] {len(out)} segments")
PYEOF
    WHISPER_OUTPUT_DIR="$OUT/04_fuzz_merged" \
    PYANNOTE_OUTPUT_DIR="$OUT/05_pyannote_diarization" \
    GLOBAL_MAP_PATH="$OUT/07_speaker_clusters/global_map.json" \
    MAIN_SPEAKER_NAME="Vaisesika Dasa" \
    AUDIO_ROOT="/lab/kiran/tbv_mp3" \
        $PYTHON 06_postprocessing/speaker_merge.py "$stem" >> "$OUT/tbv_logs/$stem.refuzz.log" 2>&1
    if [ -s "v2/$stem.json" ]; then
        mv "v2/$stem.json" "$OUT/08_final_json/$stem.json"
        echo "[DONE] $stem"
    else
        echo "[FAIL][merge] $stem"
    fi
}
export -f process_one

ls -d "$OUT"/04_fuzz_merged/*/ | xargs -P "$WORKERS" -I{} bash -c 'process_one "$@"' _ {}

echo "--- refuzz pass done at $(date) ---"
echo "=== re-exporting ==="
$PYTHON "$REPO/06_postprocessing/export_transcripts.py"
echo "=== refuzz_corpus COMPLETE at $(date) ==="

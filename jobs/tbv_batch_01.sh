#!/usr/bin/env bash
# tbv_batch_01.sh — Batch transcript pipeline over /lab/kiran/tbv_mp3 (flat MP3 layout)
#
# Per file: gemini (reuse /lab/kiran/gemini_3.0_flash cache; transcribe gaps via API)
# + whisper (one worker per GPU) -> merge -> fuzz alignment -> speaker_merge
# -> outputs/08_final_json/<stem>.json
#
# Idempotent: every stage skips existing valid output. Per-file failures never
# kill the batch; they leave a marker in outputs/tbv_failed/<stem>.<stage>
# (delete the marker to retry). Files still being downloaded (open writer, or
# modified <90s ago) are skipped and picked up on a later pass. The script
# loops until the wget download is gone AND no pending work remains.
#
# Env: GOOGLE_API_KEY required for gemini gap-fill.
#      TBV_LIMIT=N  process at most N pending files in a single pass, then exit (test mode).
#      TBV_NO_GEMINI=1  never call the Gemini API: only process files already in
#                       the cache; files without a cached transcript are skipped.
set -uo pipefail

CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"
REPO="/lab/kiran/transcript-pipeline"
AUDIO="/lab/kiran/tbv_mp3"
GBAK="/lab/kiran/gemini_3.0_flash"
OUT="$REPO/outputs"
FAILDIR="$OUT/tbv_failed"
LOGDIR="$OUT/tbv_logs"
NUM_GPUS=4
LIMIT="${TBV_LIMIT:-0}"
NO_GEMINI="${TBV_NO_GEMINI:-0}"

cd "$REPO"
mkdir -p "$OUT/01_gemini_transcripts" "$OUT/02_whisper_transcripts" \
         "$OUT/03_whisperx_alignment" "$OUT/04_fuzz_merged" \
         "$OUT/05_pyannote_diarization" "$OUT/07_speaker_clusters" \
         "$OUT/08_final_json" "$FAILDIR" "$LOGDIR" v2

echo "=== tbv_batch_01 on $(hostname) at $(date) ==="
echo "Audio: $AUDIO | Gemini cache: $GBAK | Limit: $LIMIT"

# Fully downloaded: no process has it open, and mtime is at least 90s old.
is_complete() {
    fuser -s "$1" 2>/dev/null && return 1
    local age=$(( $(date +%s) - $(stat -c %Y "$1") ))
    [ "$age" -ge 90 ]
}

valid_json() {
    [ -s "$1" ] && $PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d' "$1" 2>/dev/null
}

has_fail_marker() {
    compgen -G "$FAILDIR/$1.*" > /dev/null
}

# --- Gemini gap-fill lane: transcribe files missing from the backup cache ---
gemini_lane() {
    local listfile="$1"
    # Read the list on fd 3: child processes (ffmpeg etc.) consume stdin,
    # which mangles the stem list if it is wired to the loop's stdin.
    while IFS= read -r -u3 stem; do
        local bak="$GBAK/$stem/$stem.json"
        valid_json "$bak" && continue
        [ -f "$FAILDIR/$stem.gemini" ] && continue
        echo "[GEMINI] $stem"
        if $PYTHON 02_transcription/gemini_transcribe.py "$AUDIO/$stem.mp3" \
                --output_dir "$GBAK/$stem" --threads 4 \
                < /dev/null >> "$LOGDIR/$stem.gemini.log" 2>&1 \
           && valid_json "$bak"; then
            echo "[GEMINI-DONE] $stem"
        else
            echo "[FAIL][gemini] $stem"
            touch "$FAILDIR/$stem.gemini"
        fi
    done 3< "$listfile"
}

# --- Convert backup gemini JSON -> pipeline format (01_gemini_transcripts) ---
convert_gemini() {
    local listfile="$1"
    $PYTHON - "$GBAK" "$OUT/01_gemini_transcripts" "$FAILDIR" "$listfile" <<'PYEOF'
import sys, json, os
gbak, outbase, faildir, listfile = sys.argv[1:5]
for stem in open(listfile).read().splitlines():
    src = os.path.join(gbak, stem, stem + ".json")
    dstdir = os.path.join(outbase, stem)
    dst = os.path.join(dstdir, "transcript.json")
    if os.path.exists(dst) or not os.path.exists(src):
        continue
    try:
        with open(src) as f:
            data = json.load(f)
        if isinstance(data, dict) and "segments" in data:
            segs = [{"speaker": s.get("label", "LECTURE"), "text": s["text"],
                     "start": s["start"], "end": s["end"]}
                    for s in data["segments"]
                    if s.get("label", "LECTURE") == "LECTURE" and s.get("text", "").strip()]
        else:
            segs = [s for s in data if s.get("text", "").strip()]
        if not segs:
            open(os.path.join(faildir, stem + ".nolecture"), "w").close()
            print(f"[NOLECTURE] {stem}")
            continue
        os.makedirs(dstdir, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(segs, f, indent=2, ensure_ascii=False)
        print(f"[CONVERT] {stem}: {len(segs)} segments")
    except Exception as e:
        open(os.path.join(faildir, stem + ".convert"), "w").close()
        print(f"[FAIL][convert] {stem}: {e}")
PYEOF
}

# --- Per-file GPU stages: whisper -> merge -> fuzz -> speaker_merge ---
process_stem() {
    local stem="$1" gpu="$2"
    local mp3="$AUDIO/$stem.mp3"
    local gjson="$OUT/01_gemini_transcripts/$stem/transcript.json"
    local wdir="$OUT/02_whisper_transcripts/$stem"
    local wjson="$wdir/transcript.json"
    local adir="$OUT/03_whisperx_alignment/$stem"
    local merged="$adir/segments_with_whisper_and_gemini.json"
    local fdir="$OUT/04_fuzz_merged/$stem"
    local ftrans="$fdir/transcript.json"
    local final="$OUT/08_final_json/$stem.json"

    [ -s "$final" ] && return 0

    if ! valid_json "$wjson"; then
        [ -f "$FAILDIR/$stem.whisper" ] && return 0
        echo "[GPU$gpu][WHISPER] $stem"
        if ! CUDA_VISIBLE_DEVICES="$gpu" $PYTHON 02_transcription/whisper_transcribe.py \
                "$mp3" --output_dir "$wdir" >> "$LOGDIR/$stem.whisper.log" 2>&1 \
           || ! valid_json "$wjson"; then
            echo "[FAIL][whisper] $stem"
            touch "$FAILDIR/$stem.whisper"
            return 0
        fi
    fi

    # Gemini transcript not converted yet (gap-fill still running) -> later pass
    valid_json "$gjson" || { echo "[WAIT][gemini] $stem"; return 0; }

    if ! valid_json "$merged"; then
        [ -f "$FAILDIR/$stem.merge" ] && return 0
        mkdir -p "$adir"
        if ! $PYTHON 03_timestamp_alignment/merge_transcripts.py \
                "$wjson" "$gjson" "$merged" >> "$LOGDIR/$stem.align.log" 2>&1 \
           || ! valid_json "$merged"; then
            echo "[FAIL][merge] $stem"
            touch "$FAILDIR/$stem.merge"
            return 0
        fi
    fi

    if ! valid_json "$ftrans"; then
        [ -f "$FAILDIR/$stem.fuzz" ] && return 0
        mkdir -p "$fdir"
        cp "$merged" "$fdir/segments_with_whisper_and_gemini.json"
        if ! { $PYTHON 03_timestamp_alignment/fuzz.py "$fdir/" \
               && $PYTHON 03_timestamp_alignment/correct_gemini_timesteps.py "$fdir/" \
               && $PYTHON 03_timestamp_alignment/greedypushing_postprocess.py \
                      "$fdir/segments_corrected.json" "$fdir/segments_pushed.json"; \
             } >> "$LOGDIR/$stem.align.log" 2>&1; then
            echo "[FAIL][fuzz] $stem"
            touch "$FAILDIR/$stem.fuzz"
            return 0
        fi
        $PYTHON - "$fdir/segments_pushed.json" "$ftrans" <<'PYEOF' >> "$LOGDIR/$stem.align.log" 2>&1
import sys, json
with open(sys.argv[1]) as f:
    segs = json.load(f)
out = []
for seg in segs:
    # WHISPER_GROUP items are the raw Gemini segments whose text fuzz already
    # mapped onto WHISPER_SEGMENTs — including them duplicates every passage
    # (bug found 2026-07-03: median 2x transcript length corpus-wide).
    if seg.get("type") == "WHISPER_GROUP":
        continue
    text = (seg.get("gemini_transcript") or seg.get("whisper_transcript") or "").strip()
    if not text:
        continue
    out.append({"start": seg.get("start", 0.0), "end": seg.get("end", 0.0), "text": text})
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"[DONE] {len(out)} segments -> {sys.argv[2]}")
PYEOF
        if ! valid_json "$ftrans"; then
            echo "[FAIL][fuzz-empty] $stem"
            touch "$FAILDIR/$stem.fuzz"
            return 0
        fi
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

gpu_worker() {
    local gpu="$1" listfile="$2"
    local i=0
    # fd 3 for the list; see gemini_lane
    while IFS= read -r -u3 stem; do
        i=$((i + 1))
        [ $(( (i - 1) % NUM_GPUS )) -ne "$gpu" ] && continue
        process_stem "$stem" "$gpu" < /dev/null
    done 3< "$listfile"
}

downloads_running() {
    pgrep -f "wget.*tbv2-storage" > /dev/null 2>&1
}

# ============================== Main loop ==============================
PASS=0
while true; do
    PASS=$((PASS + 1))
    PENDING_LIST=$(mktemp /tmp/tbv_pending.XXXXXX)

    for f in "$AUDIO"/*.mp3; do
        [ -e "$f" ] || continue
        stem="$(basename "$f" .mp3)"
        [ -s "$OUT/08_final_json/$stem.json" ] && continue
        has_fail_marker "$stem" && continue
        is_complete "$f" || continue
        # No-gemini mode: only files that already have a cached transcript
        [ "$NO_GEMINI" -eq 1 ] && ! [ -s "$GBAK/$stem/$stem.json" ] && continue
        echo "$stem"
    done > "$PENDING_LIST"

    if [ "$LIMIT" -gt 0 ]; then
        head -n "$LIMIT" "$PENDING_LIST" > "$PENDING_LIST.lim"
        mv "$PENDING_LIST.lim" "$PENDING_LIST"
    fi

    N_PENDING=$(wc -l < "$PENDING_LIST")
    echo ""
    echo "=== Pass $PASS: $N_PENDING pending files at $(date) ==="

    if [ "$N_PENDING" -eq 0 ]; then
        rm -f "$PENDING_LIST"
        if downloads_running; then
            echo "No pending work; downloads still running — sleeping 120s"
            sleep 120
            continue
        fi
        echo "No pending work and downloads finished — exiting"
        break
    fi

    convert_gemini "$PENDING_LIST"

    GEMINI_PID=""
    if [ "$NO_GEMINI" -ne 1 ]; then
        gemini_lane "$PENDING_LIST" &
        GEMINI_PID=$!
    fi

    WORKER_PIDS=()
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        gpu_worker "$gpu" "$PENDING_LIST" &
        WORKER_PIDS+=($!)
    done
    for pid in "${WORKER_PIDS[@]}"; do wait "$pid"; done
    [ -n "$GEMINI_PID" ] && wait "$GEMINI_PID"

    # Newly gap-filled gemini outputs become convertible now
    convert_gemini "$PENDING_LIST"
    rm -f "$PENDING_LIST"

    echo "--- Pass $PASS summary ---"
    echo "Final JSON:   $(ls "$OUT/08_final_json" 2>/dev/null | wc -l)"
    echo "Failures:     $(ls "$FAILDIR" 2>/dev/null | wc -l)  (see $FAILDIR)"
    echo "MP3s on disk: $(ls "$AUDIO"/*.mp3 2>/dev/null | wc -l)"

    [ "$LIMIT" -gt 0 ] && { echo "TBV_LIMIT set — exiting after one pass"; break; }
done

echo ""
echo "=== tbv_batch_01 DONE at $(date) ==="
echo "Final JSON: $(ls "$OUT/08_final_json" | wc -l) | Failures: $(ls "$FAILDIR" | wc -l)"

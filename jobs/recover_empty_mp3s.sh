#!/usr/bin/env bash
# Recover 0-byte (truncated) MP3 downloads in /lab/kiran/tbv_mp3.
# Source bucket (public): https://storage.googleapis.com/tbv2-storage/fts-whisper-archives/audio/<name>
# Strategy: for each empty file, percent-encode the basename, fetch to a .part
# temp, verify it is non-empty, then atomically move into place. Never destroy
# an existing good file. Logs one line per file.
set -uo pipefail

MP3_DIR="/lab/kiran/tbv_mp3"
BASE_URL="https://storage.googleapis.com/tbv2-storage/fts-whisper-archives/audio"
LOG="/lab/kiran/transcript-pipeline/outputs/recover_empty_mp3s.log"
LIST="/lab/kiran/transcript-pipeline/outputs/recover_empty_mp3s.list"
PARALLEL=4

mkdir -p "$MP3_DIR"
: > "$LOG"

echo "=== recover_empty_mp3s START $(date) ===" | tee -a "$LOG"

# Snapshot the current set of empty files so the worklist is stable.
# Null-delimited: filenames contain single quotes, commas, unicode, '!' etc.
find "$MP3_DIR" -name '*.mp3' -size 0 -print0 > "$LIST"
TOTAL=$(tr -cd '\0' < "$LIST" | wc -c)
echo "Empty MP3s to recover: $TOTAL" | tee -a "$LOG"

fetch_one() {
    local path="$1"
    local fn enc url part code sz
    fn=$(basename "$path")
    enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$fn")
    url="$BASE_URL/$enc"
    part="$path.part"
    # -f: fail on HTTP errors; retry transient failures.
    code=$(curl -sS -f -L --retry 4 --retry-delay 3 --max-time 600 \
                -o "$part" -w '%{http_code}' "$url" 2>>"$LOG") || code="curlerr"
    if [[ "$code" == "200" && -s "$part" ]]; then
        sz=$(stat -c %s "$part")
        mv -f "$part" "$path"
        echo "[OK]   $fn  ($((sz/1000)) KB)" | tee -a "$LOG"
    else
        rm -f "$part"
        echo "[FAIL] $fn  (http=$code)" | tee -a "$LOG"
    fi
}
export -f fetch_one
export BASE_URL LOG

# Run with bounded parallelism (null-delimited to survive quotes/special chars).
xargs -0 -P "$PARALLEL" -I{} bash -c 'fetch_one "$@"' _ {} < "$LIST"

OK=$(grep -c '^\[OK\]'   "$LOG")
FAIL=$(grep -c '^\[FAIL\]' "$LOG")
STILL=$(find "$MP3_DIR" -name '*.mp3' -size 0 | wc -l)
echo "=== recover_empty_mp3s DONE $(date) ===" | tee -a "$LOG"
echo "Recovered: $OK | Failed: $FAIL | Still empty: $STILL" | tee -a "$LOG"

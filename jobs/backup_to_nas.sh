#!/usr/bin/env bash
# backup_to_nas.sh — Copy pipeline code + data to the user's home NAS.
# RUN ON igpu10 (only machine with the home key). Destination:
# kiran@23.124.116.250:/mnt/vani-nas/transcript-pipeline/
#
# Order: repo+caches (small, irreplaceable) -> audio -> outputs (waits for
# any running refuzz to finish so the copy is consistent). Resumable (-aP).
set -uo pipefail

KEY="/data/unfixed_USCILab3D/.ssh/homekey"
KH="/data/unfixed_USCILab3D/.ssh/known_hosts"
SSH="ssh -i $KEY -o UserKnownHostsFile=$KH -o BatchMode=yes"
DEST="kiran@23.124.116.250:/mnt/vani-nas/transcript-pipeline"
REPO="/lab/kiran/transcript-pipeline"
RS="rsync -a --partial --info=stats1 -e"

echo "=== backup_to_nas from $(hostname) at $(date) ==="
$SSH kiran@23.124.116.250 "mkdir -p /mnt/vani-nas/transcript-pipeline"

echo "--- [1/4] repo (code + git history) ---"
$RS "$SSH" --exclude __pycache__ --exclude 'outputs' --exclude 'export' \
    --exclude transcript_pipeline.zip --exclude 'v2' \
    "$REPO/" "$DEST/repo/"

echo "--- [2/4] gemini caches ---"
for d in gemini_3.0_flash gemini_3.0_flash_bak gemini_2.5_flash gemini_2.5_flash_bak \
         gemini_2.5_flash_lite gemini_3.1_flash_lite gemini_3.5_flash; do
    [ -d "/lab/kiran/$d" ] && $RS "$SSH" "/lab/kiran/$d" "$DEST/gemini_caches/"
done

echo "--- [3/4] source audio (62G) at $(date) ---"
$RS "$SSH" /lab/kiran/tbv_mp3 "$DEST/"

echo "--- [4/4] outputs (waiting for refuzz to finish first) ---"
while pgrep -f 'refuzz_corpus.[s]h' >/dev/null 2>&1 || \
      ! grep -q 'refuzz_corpus COMPLETE' "$REPO/outputs/refuzz_corpus.log" 2>/dev/null; do
    sleep 120
done
echo "    refuzz complete; syncing outputs + export at $(date)"
$RS "$SSH" --exclude '*.nohup' "$REPO/outputs" "$DEST/"
$RS "$SSH" "$REPO/export" "$DEST/"

echo "=== backup_to_nas COMPLETE at $(date) ==="

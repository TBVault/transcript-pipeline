#!/usr/bin/env bash
# Job: Rip audio from Govardhan 2025 YouTube playlist
# Playlist: PLkKtsBboQFvzgVeHleVX-uyXl8gCG_6NF
# Items: 9-95 (87 lectures)
# Output: /lab/kiran/govardhan/
set -euo pipefail

echo "============================================"
echo "  AUDIO RIP — Govardhan 2025 Playlist"
echo "  $(date)"
echo "============================================"

OUTPUT_DIR="/lab/kiran/govardhan"
PLAYLIST_URL="https://www.youtube.com/playlist?list=PLkKtsBboQFvzgVeHleVX-uyXl8gCG_6NF"

mkdir -p "$OUTPUT_DIR"

echo ">>> Output directory: $OUTPUT_DIR"
echo ">>> Playlist items: 9-95"
echo ""

yt-dlp \
  --playlist-items 9-95 \
  --extract-audio \
  --audio-format mp3 \
  --audio-quality 0 \
  --output "$OUTPUT_DIR/%(playlist_index)03d_%(title)s.%(ext)s" \
  --restrict-filenames \
  --no-overwrites \
  --sleep-interval 2 \
  --max-sleep-interval 5 \
  --retries 3 \
  --verbose \
  "$PLAYLIST_URL"

echo ""
echo ">>> Download complete. Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR/" | head -100
echo ""
echo ">>> Total files:"
find "$OUTPUT_DIR" -name "*.mp3" -type f | wc -l
echo ""
echo ">>> Total size:"
du -sh "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "  AUDIO RIP COMPLETE — $(date)"
echo "============================================"

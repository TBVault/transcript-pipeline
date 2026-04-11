#!/usr/bin/env bash
# Clean /dev/shm temp files from pipeline runs
echo "Cleaning /dev/shm..."
rm -f /dev/shm/tmp_dia_*.wav /dev/shm/tmp_emb_*.wav /dev/shm/tmp_gemini_*.mp3
echo "Done."

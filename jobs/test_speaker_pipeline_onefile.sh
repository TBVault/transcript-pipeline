#!/usr/bin/env bash
# Single-file end-to-end validation of the speaker-identity stage:
#   diarize -> embed+cluster -> speaker_merge -> final JSON
# Proves the chain produces real multi-speaker output before the full run.
set -euo pipefail

REPO="/lab/kiran/transcript-pipeline"
AUDIO="/lab/kiran/tbv_mp3"
OUT="$REPO/outputs"
PYTHON="/home3/kiran/anaconda3/envs/vdabase/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

STEM="$1"
echo "### STEM: $STEM"

WORK="$(mktemp -d /tmp/spk_test.XXXXXX)"
mkdir -p "$WORK/audio" "$WORK/pyannote"
ln -sf "$AUDIO/$STEM.mp3" "$WORK/audio/$STEM.mp3"
mkdir -p "$OUT/05_pyannote_diarization" "$OUT/07_speaker_clusters"

echo; echo "### [1/3] DIARIZE"
AUDIO_ROOT="$WORK/audio" PYANNOTE_OUTPUT_DIR="$WORK/pyannote" \
    "$PYTHON" "$REPO/04_diarization/diarization_from_segments_local.py"
cp "$WORK/pyannote/$STEM.txt" "$OUT/05_pyannote_diarization/$STEM.txt"
echo "  pyannote turns: $(wc -l < "$WORK/pyannote/$STEM.txt"), distinct speakers: \
$(grep -oE 'speaker_\w+' "$WORK/pyannote/$STEM.txt" | sort -u | wc -l)"

echo; echo "### [2/3] EMBED + GLOBAL CLUSTER"
GMAP="$WORK/global_map.json"
MAIN_SPEAKER_NAME="Vaisesika Dasa" \
    "$PYTHON" "$REPO/05_speaker_identity/embed_and_cluster_diarized.py" \
    "$WORK/pyannote" "$AUDIO" "$GMAP"
echo "  global_map:"; "$PYTHON" -m json.tool "$GMAP" | sed 's/^/    /'

echo; echo "### [3/3] SPEAKER_MERGE -> final JSON"
mkdir -p "$WORK/final"
WHISPER_OUTPUT_DIR="$OUT/04_fuzz_merged" \
PYANNOTE_OUTPUT_DIR="$OUT/05_pyannote_diarization" \
GLOBAL_MAP_PATH="$GMAP" \
MAIN_SPEAKER_NAME="Vaisesika Dasa" \
AUDIO_ROOT="$AUDIO" \
    "$PYTHON" "$REPO/06_postprocessing/speaker_merge.py" "$STEM"
# speaker_merge writes to v2/<stem>.json by default
RESULT="$REPO/v2/$STEM.json"
echo; echo "### RESULT speaker breakdown:"
"$PYTHON" - "$RESULT" <<'PY'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1]))
c = Counter(k for seg in d for k in seg)
print("   blocks:", len(d), " speakers:", dict(c))
PY
echo; echo "### sample blocks (first 8):"
"$PYTHON" - "$RESULT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for seg in d[:8]:
    for spk, v in seg.items():
        print(f"   [{spk}] {v['start']}-{v['end']}  {v['text'][:70]!r}")
PY
echo; echo "WORKDIR (kept for inspection): $WORK"
echo "RESULT: $RESULT"

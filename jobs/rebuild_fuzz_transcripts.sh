#!/usr/bin/env bash
# rebuild_fuzz_transcripts.sh — Fix the WHISPER_GROUP duplication bug
# (2026-07-03) in every Gemini-fused transcript.
#
# The fuzz-final builder concatenated text from BOTH the whisper segments
# (with Gemini text mapped on) AND the raw Gemini WHISPER_GROUP items,
# duplicating ~every passage (median 2x length vs the Gemini source).
# Whisper-only files (no segments_pushed.json / WHISPER_ONLY marker) are
# unaffected and skipped.
#
# Per affected stem: rebuild 04_fuzz_merged/<stem>/transcript.json from the
# existing segments_pushed.json (no realignment needed) -> re-run
# speaker_merge -> outputs/08_final_json/<stem>.json. Then re-export.
set -uo pipefail

CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"
REPO="/lab/kiran/transcript-pipeline"
OUT="$REPO/outputs"
cd "$REPO"

echo "=== rebuild_fuzz_transcripts at $(date) ==="
$PYTHON - <<'PYEOF'
import json, glob, os, subprocess, sys
REPO = '/lab/kiran/transcript-pipeline'
OUT = f'{REPO}/outputs'
ENV = dict(os.environ,
           WHISPER_OUTPUT_DIR=f'{OUT}/04_fuzz_merged',
           PYANNOTE_OUTPUT_DIR=f'{OUT}/05_pyannote_diarization',
           GLOBAL_MAP_PATH=f'{OUT}/07_speaker_clusters/global_map.json',
           MAIN_SPEAKER_NAME='Vaisesika Dasa',
           AUDIO_ROOT='/lab/kiran/tbv_mp3')
paths = sorted(glob.glob(f'{OUT}/04_fuzz_merged/*/segments_pushed.json'))
print(f'affected stems: {len(paths)}', flush=True)
done = fail = 0
for p in paths:
    d = os.path.dirname(p)
    stem = os.path.basename(d)
    if os.path.exists(f'{d}/WHISPER_ONLY'):
        continue
    try:
        segs = json.load(open(p))
        out = []
        for s in segs:
            if s.get('type') == 'WHISPER_GROUP':
                continue
            t = (s.get('gemini_transcript') or s.get('whisper_transcript') or '').strip()
            if t:
                out.append({'start': s.get('start', 0.0), 'end': s.get('end', 0.0), 'text': t})
        if not out:
            fail += 1; continue
        json.dump(out, open(f'{d}/transcript.json', 'w'), indent=2, ensure_ascii=False)
        r = subprocess.run([sys.executable, f'{REPO}/06_postprocessing/speaker_merge.py', stem],
                           env=ENV, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        v2 = f'{REPO}/v2/{stem}.json'
        if r.returncode == 0 and os.path.exists(v2) and os.path.getsize(v2) > 2:
            os.replace(v2, f'{OUT}/08_final_json/{stem}.json')
            done += 1
        else:
            fail += 1
            print(f'[FAIL] {stem}', flush=True)
    except Exception as e:
        fail += 1
        print(f'[FAIL] {stem}: {e}', flush=True)
    if (done + fail) % 200 == 0:
        print(f'  ...{done} rebuilt, {fail} failed', flush=True)
print(f'rebuild done: {done} rebuilt, {fail} failed', flush=True)
PYEOF

echo "=== re-exporting changed transcripts ==="
$PYTHON "$REPO/06_postprocessing/export_transcripts.py"
echo "=== rebuild_fuzz_transcripts COMPLETE at $(date) ==="

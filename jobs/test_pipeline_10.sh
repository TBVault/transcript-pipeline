#!/usr/bin/env bash
# test_pipeline_10.sh — End-to-end pipeline on 10 test MP3 files
# Picks 10 MP3s that already have Gemini transcripts in the backup dir,
# runs all stages, writes outputs to /lab/kiran/transcript-pipeline/outputs/
set -euo pipefail

REPO_DIR="$HOME/transcript-pipeline"
AUDIO_ROOT="/dev/shm/organized_mp3"
GEMINI_BAK="/lab/kiran/gemini_3.0_flash_bak"
BASE_OUT="/lab/kiran/transcript-pipeline/outputs"
N_TEST=10

echo "=== test_pipeline_10 on $(hostname) at $(date) ==="
cd "$REPO_DIR"

# ---- Pull latest repo changes ----
git pull --ff-only 2>/dev/null || true

# ---- Ensure output dirs exist ----
for d in 01_gemini_transcripts 02_whisper_transcripts 03_whisperx_alignment \
          04_fuzz_merged 05_pyannote_diarization 06_wavlm_embeddings \
          07_speaker_clusters 08_final_json; do
    mkdir -p "$BASE_OUT/$d"
done

# ---- Select 10 MP3 folders that have existing Gemini transcripts ----
echo ""
echo "[0/7] Selecting $N_TEST test files..."
TEST_FILES=()
count=0
for folder in "$AUDIO_ROOT"/*/; do
    fname=$(basename "$folder")
    if [ -d "$GEMINI_BAK/$fname" ]; then
        TEST_FILES+=("$fname")
        count=$((count+1))
        if [ "$count" -ge "$N_TEST" ]; then break; fi
    fi
done
echo "Selected $count files:"
for f in "${TEST_FILES[@]}"; do echo "  $f"; done

# ---- Stage 1: Convert existing Gemini transcripts to pipeline format ----
echo ""
echo "[1/7] Converting Gemini backup transcripts..."
python3 - "${TEST_FILES[@]}" <<'PYEOF'
import sys, json, os

fnames = sys.argv[1:]
gemini_bak = "/lab/kiran/gemini_3.0_flash_bak"
out_base   = "/lab/kiran/transcript-pipeline/outputs/01_gemini_transcripts"

for fname in fnames:
    src = os.path.join(gemini_bak, fname, fname + ".json")
    dst_dir = os.path.join(out_base, fname)
    dst = os.path.join(dst_dir, "transcript.json")
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(dst):
        print(f"[SKIP] {fname}")
        continue
    with open(src) as f:
        data = json.load(f)
    # Convert from legacy {file_id, segments:[{label, start, end, text}]}
    # to pipeline format [{speaker, text, start, end}]
    if isinstance(data, dict) and "segments" in data:
        segs = [
            {"speaker": s.get("label", "LECTURE"), "text": s["text"],
             "start": s["start"], "end": s["end"]}
            for s in data["segments"]
            if s.get("label", "LECTURE") == "LECTURE" and s.get("text", "").strip()
        ]
    else:
        segs = [s for s in data if s.get("text", "").strip()]
    with open(dst, "w") as f:
        json.dump(segs, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {fname}: {len(segs)} lecture segments")
PYEOF

# ---- Stage 2: Whisper transcription ----
echo ""
echo "[2/7] Running Whisper transcription (GPU 0)..."
export CUDA_VISIBLE_DEVICES="0"
for fname in "${TEST_FILES[@]}"; do
    mp3="$AUDIO_ROOT/$fname/$fname.mp3"
    out_dir="$BASE_OUT/02_whisper_transcripts/$fname"
    out_file="$out_dir/transcript.json"
    if [ -f "$out_file" ]; then
        echo "[SKIP] whisper $fname"
        continue
    fi
    echo "  Transcribing: $fname"
    python 02_transcription/whisper_transcribe.py "$mp3" --output_dir "$out_dir"
done

# ---- Stage 3: Merge Whisper + Gemini → alignment input ----
echo ""
echo "[3/7] Merging Whisper + Gemini transcripts..."
for fname in "${TEST_FILES[@]}"; do
    whisper_json="$BASE_OUT/02_whisper_transcripts/$fname/transcript.json"
    gemini_json="$BASE_OUT/01_gemini_transcripts/$fname/transcript.json"
    align_dir="$BASE_OUT/03_whisperx_alignment/$fname"
    merged_out="$align_dir/segments_with_whisper_and_gemini.json"
    if [ -f "$merged_out" ]; then
        echo "[SKIP] merge $fname"
        continue
    fi
    if [ ! -f "$whisper_json" ]; then
        echo "[WARN] whisper missing for $fname — skipping merge"
        continue
    fi
    echo "  Merging: $fname"
    mkdir -p "$align_dir"
    python 03_timestamp_alignment/merge_transcripts.py \
        "$whisper_json" "$gemini_json" "$merged_out"
done

# ---- Stage 4: Fuzz + timestamp correction + greedy push ----
echo ""
echo "[4/7] Fuzz alignment + timestamp correction..."
for fname in "${TEST_FILES[@]}"; do
    align_dir="$BASE_OUT/03_whisperx_alignment/$fname"
    fuzz_dir="$BASE_OUT/04_fuzz_merged/$fname"
    merged="$align_dir/segments_with_whisper_and_gemini.json"
    final_transcript="$fuzz_dir/transcript.json"

    if [ ! -f "$merged" ]; then
        echo "[SKIP] fuzz $fname (no merged file)"
        continue
    fi
    if [ -f "$final_transcript" ]; then
        echo "[SKIP] fuzz $fname (already done)"
        continue
    fi
    mkdir -p "$fuzz_dir"
    echo "  Fuzz: $fname"

    # fuzz.py reads segments_with_whisper_and_gemini.json from a dir
    # — copy to fuzz dir, run fuzz, correct, greedy push
    cp "$merged" "$fuzz_dir/segments_with_whisper_and_gemini.json"
    python 03_timestamp_alignment/fuzz.py "$fuzz_dir/"
    python 03_timestamp_alignment/correct_gemini_timesteps.py "$fuzz_dir/"
    python 03_timestamp_alignment/greedypushing_postprocess.py \
        "$fuzz_dir/segments_corrected.json" \
        "$fuzz_dir/segments_pushed.json"

    # Convert to transcript.json format for speaker_merge.py:
    # [{start, end, text}] using gemini_transcript where available
    python3 - "$fuzz_dir/segments_pushed.json" "$final_transcript" <<'PYEOF'
import sys, json

with open(sys.argv[1]) as f:
    segs = json.load(f)

out = []
for seg in segs:
    t = seg.get("type", "")
    text = (seg.get("gemini_transcript") or seg.get("whisper_transcript") or "").strip()
    if not text:
        continue
    out.append({"start": seg.get("start", 0.0), "end": seg.get("end", 0.0), "text": text})

with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"[DONE] {len(out)} segments -> {sys.argv[2]}")
PYEOF
done

# ---- Stage 5: PyAnnote diarization (multi-GPU parallel) ----
echo ""
echo "[5/7] PyAnnote diarization (4 V100 GPUs)..."
export HF_TOKEN="${HF_TOKEN:-}"
python3 - "${TEST_FILES[@]}" <<'PYEOF'
import sys, os, subprocess
from multiprocessing import Process
import torch

fnames = sys.argv[1:]
AUDIO_ROOT = "/dev/shm/organized_mp3"
BASE_OUT = "/lab/kiran/transcript-pipeline/outputs"
PYANNOTE_OUT = f"{BASE_OUT}/05_pyannote_diarization"
VOICEPRINTS = f"{BASE_OUT}/06_wavlm_embeddings"
HF_TOKEN = os.getenv("HF_TOKEN", "")
NUM_GPUS = max(torch.cuda.device_count(), 1)

pending = [f for f in fnames if not os.path.exists(f"{PYANNOTE_OUT}/{f}.txt")]
if not pending:
    print("[SKIP] all diarization already done")
    sys.exit(0)

print(f"Diarizing {len(pending)} files on {NUM_GPUS} GPUs")
os.makedirs(PYANNOTE_OUT, exist_ok=True)

def run_one(fname, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["AUDIO_ROOT"] = os.path.join(AUDIO_ROOT, fname)
    env["PYANNOTE_OUTPUT_DIR"] = PYANNOTE_OUT
    env["VOICEPRINTS_DIR"] = VOICEPRINTS
    env["HF_TOKEN"] = HF_TOKEN
    result = subprocess.run(
        ["python", "04_diarization/split_audio_dia.py", fname],
        env=env, capture_output=True, text=True, cwd=os.path.expanduser("~/transcript-pipeline")
    )
    out = (result.stdout + result.stderr).strip()
    print(f"[GPU {gpu_id}] {fname[:60]}: {out[-300:]}")

active, i = [], 0
for i, fname in enumerate(pending):
    gpu = i % NUM_GPUS
    p = Process(target=run_one, args=(fname, gpu))
    p.start()
    active.append(p)
    if len(active) >= NUM_GPUS:
        active[0].join()
        active.pop(0)

for p in active:
    p.join()
print("[DONE] diarization complete")
PYEOF

# ---- Stage 6: Global speaker clustering ----
echo ""
echo "[6/7] Global speaker clustering..."
GLOBAL_MAP="$BASE_OUT/07_speaker_clusters/global_map.json"
EMB_DIR="$BASE_OUT/06_wavlm_embeddings"
if [ ! -f "$GLOBAL_MAP" ]; then
    python 05_speaker_identity/global_clustering.py "$EMB_DIR" "$GLOBAL_MAP"
else
    echo "[SKIP] global_map.json exists"
fi

# ---- Stage 7: Speaker merge → final JSON ----
echo ""
echo "[7/7] Speaker merge → final JSON..."
export WHISPER_OUTPUT_DIR="$BASE_OUT/04_fuzz_merged"
export PYANNOTE_OUTPUT_DIR="$BASE_OUT/05_pyannote_diarization"
export GLOBAL_MAP_PATH="$BASE_OUT/07_speaker_clusters/global_map.json"
export MAIN_SPEAKER_NAME="Vaisesika Dasa"
export AUDIO_ROOT="$AUDIO_ROOT"

for fname in "${TEST_FILES[@]}"; do
    final_out="$BASE_OUT/08_final_json/$fname.json"
    if [ -f "$final_out" ]; then
        echo "[SKIP] speaker_merge $fname"
        continue
    fi
    whisper_seg="$BASE_OUT/04_fuzz_merged/$fname/transcript.json"
    diar_txt="$BASE_OUT/05_pyannote_diarization/$fname.txt"
    if [ ! -f "$whisper_seg" ]; then
        echo "[WARN] missing fuzz output for $fname"
        continue
    fi
    echo "  Speaker merge: $fname"
    # Run speaker_merge.py via env vars (it reads WHISPER_OUTPUT_DIR/{lid}/transcript.json)
    python 06_postprocessing/speaker_merge.py "$fname" 2>&1 | tail -1
    # Move default output to our output dir
    if [ -f "v2/$fname.json" ]; then
        mv "v2/$fname.json" "$final_out"
    fi
done

# ---- Summary ----
echo ""
echo "=== Pipeline complete at $(date) ==="
echo "Stage 1  Gemini transcripts:  $(ls $BASE_OUT/01_gemini_transcripts/ 2>/dev/null | wc -l) folders"
echo "Stage 2  Whisper transcripts: $(ls $BASE_OUT/02_whisper_transcripts/ 2>/dev/null | wc -l) folders"
echo "Stage 3  Merged alignment:    $(ls $BASE_OUT/03_whisperx_alignment/  2>/dev/null | wc -l) folders"
echo "Stage 4  Fuzz output:         $(ls $BASE_OUT/04_fuzz_merged/          2>/dev/null | wc -l) folders"
echo "Stage 5  Diarization:         $(ls $BASE_OUT/05_pyannote_diarization/ 2>/dev/null | wc -l) files"
echo "Stage 6  WavLM embeddings:    $(ls $BASE_OUT/06_wavlm_embeddings/     2>/dev/null | wc -l) folders"
echo "Stage 7  Speaker clusters:    $(ls $BASE_OUT/07_speaker_clusters/     2>/dev/null | wc -l) files"
echo "Stage 8  Final JSON:          $(ls $BASE_OUT/08_final_json/           2>/dev/null | wc -l) files"
echo ""
echo "Sample final output:"
ls "$BASE_OUT/08_final_json/" 2>/dev/null | head -3 | while read f; do
    echo "  $f: $(python3 -c "import json; d=json.load(open('$BASE_OUT/08_final_json/$f')); print(len(d), 'blocks')" 2>/dev/null || echo '?')"
done

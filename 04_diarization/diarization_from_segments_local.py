"""
diarization_from_segments_local.py - Multi-GPU Batch PyAnnote Diarization

Each worker gets a GPU and processes files from a shared queue.

Usage: python diarization_from_segments_local.py
Environment: HF_TOKEN, AUDIO_ROOT, PYANNOTE_OUTPUT_DIR
"""
import os, random, torch
import multiprocessing as mp
from pathlib import Path
from pyannote.audio import Pipeline

FOLDER = os.getenv("AUDIO_ROOT", ".")
OUT_DIR = os.getenv("PYANNOTE_OUTPUT_DIR", "FTS_MP3_pyannote/")
HF_TOKEN = os.getenv("HF_TOKEN", "")
NUM_GPUS = max(1, torch.cuda.device_count())

def worker(gpu_id, file_list):
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    pipe.to(torch.device(f"cuda:{gpu_id}"))
    for af, out in file_list:
        try:
            output = pipe(af)
            lines = [f"start={t.start:.1f}s stop={t.end:.1f}s speaker_{s}" for t, _, s in output.itertracks(yield_label=True)]
            with open(out, "w") as f: f.write("\n".join(lines))
            print(f"[GPU {gpu_id}] {Path(af).stem}: {len(lines)} turns")
        except Exception as e: print(f"[GPU {gpu_id}] {af}: {e}")

def main():
    files = sorted(Path(FOLDER).glob("*.mp3")); random.shuffle(files)
    os.makedirs(OUT_DIR, exist_ok=True)
    lists = [[] for _ in range(NUM_GPUS)]
    for i, af in enumerate(files):
        out = os.path.join(OUT_DIR, f"{af.stem}.txt")
        if not os.path.exists(out): lists[i % NUM_GPUS].append((str(af), out))
    procs = []
    for g in range(NUM_GPUS):
        if lists[g]:
            p = mp.Process(target=worker, args=(g, lists[g])); p.start(); procs.append(p)
    for p in procs: p.join()
    print("[DONE] All GPUs finished.")

if __name__ == "__main__": main()

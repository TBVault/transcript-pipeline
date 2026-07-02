"""
diarization_from_segments_local.py - Multi-GPU Batch PyAnnote Diarization

Each worker gets a GPU and processes files from a shared queue.

Usage: python diarization_from_segments_local.py
Environment: HF_TOKEN, AUDIO_ROOT, PYANNOTE_OUTPUT_DIR
"""
import os, random, torch
import multiprocessing as mp
from pathlib import Path

import omegaconf
import omegaconf.base, omegaconf.nodes, omegaconf.basecontainer
# PyTorch 2.6 fix: allow omegaconf globals when loading the pyannote checkpoint.
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.nodes.ValueNode,
    omegaconf.basecontainer.BaseContainer,
])
# PyTorch 2.6 ships weights_only=True by default; lightning passes it
# explicitly, so the trusted local pyannote checkpoint fails to load with
# "Weights only load failed". Force pre-2.6 behavior — same fix already in
# whisper_transcribe.py. Without it the diarization stage silently produces
# nothing and every transcript collapses to the single fallback speaker.
_torch_load = torch.load
def _load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)
torch.load = _load_trusted

from pyannote.audio import Pipeline

FOLDER = os.getenv("AUDIO_ROOT", ".")
OUT_DIR = os.getenv("PYANNOTE_OUTPUT_DIR", "FTS_MP3_pyannote/")
# Env token if provided, else fall back to the cached HF login (~/.cache/
# huggingface/token) via use_auth_token=True.
HF_TOKEN = os.getenv("HF_TOKEN", "")
NUM_GPUS = max(1, torch.cuda.device_count())
# These 8 kbps MP3s are slow to decode on CPU, so a single worker per GPU leaves
# the GPU idle (~0% util) waiting on audio. Run several workers per GPU to overlap
# CPU decode with GPU inference. pyannote needs ~2-4 GB/worker; on 32 GB V100s
# that comfortably coexists with the whisper batch. Tune via WORKERS_PER_GPU.
WORKERS_PER_GPU = int(os.getenv("WORKERS_PER_GPU", "1"))

def worker(gpu_id, file_list):
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN or True)
    pipe.to(torch.device(f"cuda:{gpu_id}"))
    for af, out in file_list:
        try:
            output = pipe(af)
            lines = [f"start={t.start:.1f}s stop={t.end:.1f}s speaker_{s}" for t, _, s in output.itertracks(yield_label=True)]
            with open(out, "w") as f: f.write("\n".join(lines))
            print(f"[GPU {gpu_id}] {Path(af).stem}: {len(lines)} turns", flush=True)
        except Exception as e: print(f"[GPU {gpu_id}] {af}: {e}", flush=True)

def main():
    files = sorted(Path(FOLDER).glob("*.mp3")); random.shuffle(files)
    os.makedirs(OUT_DIR, exist_ok=True)
    n_workers = NUM_GPUS * WORKERS_PER_GPU
    lists = [[] for _ in range(n_workers)]
    pending = 0
    for af in files:
        out = os.path.join(OUT_DIR, f"{af.stem}.txt")
        if not os.path.exists(out):
            lists[pending % n_workers].append((str(af), out)); pending += 1
    print(f"[diarize] {pending} pending across {n_workers} workers "
          f"({WORKERS_PER_GPU}/gpu x {NUM_GPUS} gpus)", flush=True)
    procs = []
    for w in range(n_workers):
        if lists[w]:
            gpu = w % NUM_GPUS
            p = mp.Process(target=worker, args=(gpu, lists[w])); p.start(); procs.append(p)
    for p in procs: p.join()
    print("[DONE] All GPUs finished.")

if __name__ == "__main__": main()

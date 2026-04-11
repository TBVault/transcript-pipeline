"""
split_audio_dia.py - PyAnnote 3.1 Speaker Diarization (Production)

Runs diarization on lecture audio respecting kirtan boundaries.
Extracts WavLM speaker embeddings per-segment for identity clustering.
15-minute timeout per file.

Usage: python split_audio_dia.py <lecture_id>
Environment: HF_TOKEN, CUDA_VISIBLE_DEVICES, AUDIO_ROOT
"""
import json, os, sys, gc, signal
import numpy as np, torch, torchaudio
from pyannote.audio import Pipeline, Model, Inference

FOLDER = os.getenv("AUDIO_ROOT", ".")
SEGS_DIR = os.getenv("KIRTANTIMES_DIR", "FTS_MP3_kirtantimes/")
OUT_DIR = os.getenv("PYANNOTE_OUTPUT_DIR", "FTS_MP3_pyannote/")
EMB_DIR = os.getenv("VOICEPRINTS_DIR", "FTS_MP3_voiceprints/")
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000; TIMEOUT = 900

class TimeoutErr(Exception): pass
def alarm_handler(s, f): raise TimeoutErr("Timeout")

def safe_load(path, start, end, sr=SR):
    try:
        info = torchaudio.info(path)
        sf, ef = int(start * info.sample_rate), min(int(end * info.sample_rate), info.num_frames)
        if ef - sf < 160: return None
        wav, osr = torchaudio.load(path, frame_offset=sf, num_frames=ef-sf)
        if osr != sr: wav = torchaudio.transforms.Resample(osr, sr)(wav)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        return wav
    except: return None

def main():
    lid = sys.argv[1]
    audio = os.path.join(FOLDER, f"{lid}.mp3")
    seg_path = os.path.join(SEGS_DIR, f"{lid}.json")
    out = os.path.join(OUT_DIR, f"{lid}.txt")
    edir = os.path.join(EMB_DIR, lid)
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(edir, exist_ok=True)
    if os.path.exists(out): print(f"[SKIP] {lid}"); return

    signal.signal(signal.SIGALRM, alarm_handler); signal.alarm(TIMEOUT)
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN).to(torch.device(DEVICE))
        emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN).to(torch.device(DEVICE))
        inference = Inference(emb_model, window="whole")

        segments = []
        if os.path.exists(seg_path):
            with open(seg_path) as f:
                for pair in json.load(f):
                    def ts(s):
                        p = s.split(":"); return float(p[0])*60+float(p[1]) if len(p)==2 else float(p[0])*3600+float(p[1])*60+float(p[2])
                    segments.append((ts(pair[0]), ts(pair[1])))
        else:
            info = torchaudio.info(audio); segments = [(0.0, info.num_frames / info.sample_rate)]

        results = []
        for start, end in segments:
            wav = safe_load(audio, start, end)
            if wav is None: continue
            tmp = f"/dev/shm/tmp_dia_{lid}_{int(start)}.wav"
            torchaudio.save(tmp, wav, SR)
            try:
                output = pipe(tmp)
                for turn, _, spk in output.itertracks(yield_label=True):
                    a_s, a_e = turn.start + start, turn.end + start
                    results.append(f"start={a_s:.1f}s stop={a_e:.1f}s speaker_{spk}")
                    sw = safe_load(audio, a_s, a_e)
                    if sw is not None:
                        st = f"/dev/shm/tmp_emb_{lid}.wav"; torchaudio.save(st, sw, SR)
                        try: np.save(os.path.join(edir, f"{int(a_s)}_{int(a_e)}.npy"), inference(st))
                        except: pass
                        if os.path.exists(st): os.remove(st)
            finally:
                if os.path.exists(tmp): os.remove(tmp)
        with open(out, "w") as f: f.write("\n".join(results))
        print(f"[DONE] {lid}: {len(results)} turns -> {out}")
    except TimeoutErr: print(f"[TIMEOUT] {lid}")
    finally: signal.alarm(0); gc.collect(); torch.cuda.is_available() and torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python split_audio_dia.py <lecture_id>"); sys.exit(1)
    main()

"""
gen_embeddings.py - WavLM Speaker Embedding Extraction

30s sliding window, 50% overlap, L2-normalized x-vectors.

Usage: python gen_embeddings.py <lecture_dir/>
"""
import os, sys, json
import numpy as np, torch, torchaudio
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

MODEL = "microsoft/wavlm-base-plus-sv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000

def main():
    folder = Path(sys.argv[1]); emb_dir = folder / "embeddings"; emb_dir.mkdir(exist_ok=True)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).to(DEVICE); model.eval()

    for json_path in folder.glob("*.json"):
        audio_path = json_path.with_suffix(".mp3")
        if not audio_path.exists(): continue
        with open(json_path) as f: data = json.load(f)
        for item in (data if isinstance(data, list) else []):
            if isinstance(item, list) and len(item) == 2:
                def ts(s):
                    p = s.split(":"); return float(p[0])*60+float(p[1]) if len(p)==2 else float(p[0])*3600+float(p[1])*60+float(p[2])
                start, end = ts(item[0]), ts(item[1])
            elif isinstance(item, dict) and item.get("label","").upper() == "LECTURE":
                start, end = item["start"], item["end"]
            else: continue
            out = emb_dir / f"{audio_path.stem}_{int(start)}_{int(end)}.npy"
            if out.exists(): continue
            chunks = []
            t, win, step = 0.0, 30.0, 15.0
            dur = end - start
            while t < dur:
                cl = min(dur - t, win)
                if cl < 1.0: break
                try:
                    info = torchaudio.info(str(audio_path))
                    sf = int((start+t)*info.sample_rate)
                    nf = min(int(cl*info.sample_rate), info.num_frames - sf)
                    wav, sr = torchaudio.load(str(audio_path), frame_offset=sf, num_frames=nf)
                    if sr != SR: wav = torchaudio.transforms.Resample(sr, SR)(wav)
                    wav = wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
                    with torch.no_grad():
                        inp = proc([wav.numpy()], sampling_rate=SR, return_tensors="pt", padding=True)
                        inp = {k: v.to(DEVICE) for k, v in inp.items()}
                        chunks.append(model(**inp).embeddings.cpu().numpy())
                except: pass
                t += step
            if chunks:
                emb = np.mean(np.vstack(chunks), axis=0)
                norm = np.linalg.norm(emb)
                if norm > 0: emb /= norm
                np.save(str(out), emb)
                print(f"  Saved {start:.0f}-{end:.0f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python gen_embeddings.py <dir>"); sys.exit(1)
    main()

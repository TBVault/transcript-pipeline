"""
embed_and_cluster_diarized.py - Diarization-aware speaker identity.

Replaces the gen_embeddings.py + global_clustering.py pair, whose output was
keyed by time-based agglomerative-cluster indices and therefore never lined up
with what speaker_merge.py looks up (the pyannote SPEAKER_xx label). This script
closes that seam:

  1. For each lecture, read the pyannote turns (05_pyannote_diarization/<stem>.txt).
  2. Per pyannote speaker (SPEAKER_00, ...), embed up to EMBED_BUDGET seconds of
     that speaker's longest turns with WavLM x-vectors -> one L2-normalized
     centroid per (lecture, SPEAKER_xx).
  3. Global agglomerative clustering (cosine) over all centroids -> persistent
     identities. The globally largest cluster (by total speaking time) is the
     MAIN_SPEAKER; the rest become "Audience N".
  4. Emit global_map.json = { "<stem>": { "SPEAKER_00": "<name>", ... }, ... },
     exactly the shape speaker_merge.py expects (smap.get(raw)).

Usage:
    python embed_and_cluster_diarized.py <pyannote_txt_dir> <audio_root> <out_map.json>
Env: MAIN_SPEAKER_NAME (default "Vaisesika Dasa")
"""
import os, sys, re, json
import numpy as np
import torch, torchaudio
from collections import defaultdict
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from sklearn.cluster import AgglomerativeClustering

MODEL = "microsoft/wavlm-base-plus-sv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
EMBED_BUDGET = 60.0          # seconds of audio to embed per speaker
WIN, STEP = 30.0, 15.0
GLOBAL_THRESH = float(os.getenv("GLOBAL_THRESH", "0.35"))
MAIN_SPEAKER = os.getenv("MAIN_SPEAKER_NAME", "Vaisesika Dasa")

TURN_RE = re.compile(r"start=([\d.]+)s\s+stop=([\d.]+)s\s+speaker_(\w+)")


def parse_turns(txt_path):
    """Return {speaker_label: [(start, end), ...]} from a pyannote .txt."""
    by_spk = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            m = TURN_RE.search(line.strip())
            if not m:
                continue
            s, e, spk = float(m[1]), float(m[2]), m[3]
            if e > s:
                by_spk[spk].append((s, e))
    return by_spk


def pick_windows(turns, budget=EMBED_BUDGET):
    """Longest turns first, accumulate up to `budget` seconds of audio."""
    chosen, total = [], 0.0
    for s, e in sorted(turns, key=lambda t: t[1] - t[0], reverse=True):
        if total >= budget:
            break
        chosen.append((s, min(e, s + (budget - total))))
        total += e - s
    return chosen


def embed_speaker(audio_path, windows, proc, model):
    """Average WavLM x-vector over the chosen windows. None if nothing usable."""
    info = torchaudio.info(str(audio_path))
    vecs = []
    for s, e in windows:
        t = 0.0
        dur = e - s
        while t < dur:
            cl = min(dur - t, WIN)
            if cl < 1.0:
                break
            try:
                sf = int((s + t) * info.sample_rate)
                nf = min(int(cl * info.sample_rate), info.num_frames - sf)
                if nf <= 0:
                    break
                wav, sr = torchaudio.load(str(audio_path), frame_offset=sf, num_frames=nf)
                if sr != SR:
                    wav = torchaudio.transforms.Resample(sr, SR)(wav)
                wav = wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
                with torch.no_grad():
                    inp = proc([wav.numpy()], sampling_rate=SR, return_tensors="pt", padding=True)
                    inp = {k: v.to(DEVICE) for k, v in inp.items()}
                    vecs.append(model(**inp).embeddings.cpu().numpy())
            except Exception:
                pass
            t += STEP
    if not vecs:
        return None
    emb = np.mean(np.vstack(vecs), axis=0)
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else None


def main():
    pyannote_dir, audio_root, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).to(DEVICE)
    model.eval()

    centroids = []   # {stem, spk, emb, dur}
    txts = sorted(f for f in os.listdir(pyannote_dir) if f.endswith(".txt"))
    # Embeddings are the only expensive part (they need the audio); clustering is
    # cheap. Cache one .npy per (lecture, speaker) so threshold sweeps and reruns
    # skip re-embedding the whole corpus. Durations come from the turns (free).
    cache_dir = os.getenv("EMB_CACHE_DIR",
                          os.path.join(os.path.dirname(out_file) or ".", "emb_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    cached = reembedded = 0
    for i, fn in enumerate(txts):
        stem = fn[:-4]
        audio = os.path.join(audio_root, stem + ".mp3")
        if not os.path.exists(audio):
            continue
        by_spk = parse_turns(os.path.join(pyannote_dir, fn))
        for spk, turns in by_spk.items():
            ckey = os.path.join(cache_dir, f"{stem}__{spk}.npy")
            if os.path.exists(ckey):
                try:
                    emb = np.load(ckey); cached += 1
                except Exception:
                    emb = None
            else:
                emb = embed_speaker(audio, pick_windows(turns), proc, model)
                if emb is not None:
                    np.save(ckey, emb); reembedded += 1
            if emb is None:
                continue
            centroids.append({"stem": stem, "spk": spk, "emb": emb,
                              "dur": sum(e - s for s, e in turns)})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(txts)} lectures, {len(centroids)} centroids "
                  f"(cached={cached} new={reembedded})", flush=True)

    if not centroids:
        print("No centroids; writing empty map.")
        with open(out_file, "w") as f:
            json.dump({}, f)
        return

    X = np.array([c["emb"] for c in centroids])
    if len(centroids) == 1:
        labels = [0]
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=GLOBAL_THRESH,
            metric="cosine", linkage="average").fit_predict(X)
    for c, l in zip(centroids, labels):
        c["global"] = int(l)

    dur_by_global = defaultdict(float)
    for c in centroids:
        dur_by_global[c["global"]] += c["dur"]
    main_cluster = max(dur_by_global, key=dur_by_global.get)

    # Stable "Audience N" numbering by descending total speaking time.
    others = sorted((g for g in dur_by_global if g != main_cluster),
                    key=lambda g: dur_by_global[g], reverse=True)
    name_of = {main_cluster: MAIN_SPEAKER}
    for n, g in enumerate(others, 1):
        name_of[g] = f"Audience {n}"

    gmap = defaultdict(dict)
    for c in centroids:
        gmap[c["stem"]][c["spk"]] = name_of[c["global"]]

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(gmap, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {len(gmap)} lectures, {len(centroids)} speaker-centroids, "
          f"{len(set(labels))} global identities -> {out_file}")
    print(f"  MAIN={MAIN_SPEAKER} ({dur_by_global[main_cluster]:.0f}s), "
          f"+{len(others)} audience identities")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python embed_and_cluster_diarized.py <pyannote_txt_dir> <audio_root> <out_map.json>")
        sys.exit(1)
    main()

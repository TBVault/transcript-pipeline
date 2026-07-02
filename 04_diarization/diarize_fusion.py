"""
diarize_fusion.py - Fusion speaker segmentation: pyannote 3.1 + DiariZen
(WavLM-Conformer, ~12.7% DER vs pyannote's ~16%), combined with DOVER-Lap
weighted voting.

Reuses the existing corpus pyannote output (outputs/05_pyannote_diarization/
<stem>.txt) when present — only DiariZen runs fresh; pass --rerun-pyannote to
force both. The two systems' turn lists are written as RTTM, fused by
DOVER-Lap (rank-weighted majority voting on speaker-mapped regions), and the
fused segmentation is emitted in the pipeline's native turn format:

    start=12.34s stop=56.78s speaker_SPEAKER_00

Run with the diarizen overlay venv (NOT vdabase):
    LD_LIBRARY_PATH=/home3/kiran/anaconda3/envs/tbv/lib \
    /lab/kiran/diarizen_venv/bin/python 04_diarization/diarize_fusion.py \
        <audio.mp3> --out_dir outputs/06_fusion_diarization

Env: DIARIZEN_MODEL (default BUT-FIT/diarizen-wavlm-large-s80-md, cached),
     PYANNOTE_TXT_DIR (default outputs/05_pyannote_diarization),
     DOVER_WEIGHTS (default "1.0,1.1" for pyannote,diarizen — slight edge to
     the stronger model; DOVER-Lap re-ranks internally anyway).
"""
import os, sys, re, argparse, tempfile, subprocess

# PyTorch >= 2.6 defaults torch.load(weights_only=True); pyannote/diarizen
# checkpoints need the old behavior (see whisper_transcribe.py).
import torch
_orig_load = torch.load
def _load(*a, **k):
    k["weights_only"] = False   # some callers pass True explicitly
    return _orig_load(*a, **k)
torch.load = _load

TURN_RE = re.compile(r"start=([\d.]+)s\s+stop=([\d.]+)s\s+speaker_(\w+)")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def turns_to_rttm(turns, uri, path):
    with open(path, "w") as f:
        for start, stop, spk in turns:
            if stop > start:
                f.write(f"SPEAKER {uri} 1 {start:.3f} {stop - start:.3f} "
                        f"<NA> <NA> {spk} <NA> <NA>\n")


def rttm_to_turns(path):
    turns = []
    for line in open(path):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            turns.append((start, start + dur, p[7]))
    return sorted(turns)


def read_pipeline_txt(path):
    turns = []
    for line in open(path):
        m = TURN_RE.search(line)
        if m and float(m[2]) > float(m[1]):
            spk = m[3] if m[3].startswith("SPEAKER") else "SPEAKER_" + m[3]
            turns.append((float(m[1]), float(m[2]), spk))
    return sorted(turns)


def annotation_to_turns(ann):
    return sorted((seg.start, seg.end, str(lbl))
                  for seg, _, lbl in ann.itertracks(yield_label=True))


def run_pyannote(wav):
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=os.getenv("HF_TOKEN") or True)
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return annotation_to_turns(pipe(wav))


def run_diarizen(wav):
    from diarizen.pipelines.inference import DiariZenPipeline
    model = os.getenv("DIARIZEN_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md")
    pipe = DiariZenPipeline.from_pretrained(model)
    return annotation_to_turns(pipe(wav))


def to_wav16k(audio, tmpdir):
    """8 kbps corpus MP3s decode unreliably in torchaudio — go through ffmpeg."""
    wav = os.path.join(tmpdir, "audio16k.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", audio, "-ac", "1",
                    "-ar", "16000", wav], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out_dir", default=os.path.join(REPO, "outputs/06_fusion_diarization"))
    ap.add_argument("--rerun-pyannote", action="store_true")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.audio))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    out_txt = os.path.join(args.out_dir, stem + ".txt")
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        print(f"[fusion] exists, skipping: {out_txt}")
        return

    pya_txt = os.path.join(os.getenv("PYANNOTE_TXT_DIR",
                           os.path.join(REPO, "outputs/05_pyannote_diarization")),
                           stem + ".txt")

    with tempfile.TemporaryDirectory(dir=os.getenv("TMPDIR", "/tmp")) as td:
        wav = to_wav16k(args.audio, td)

        if not args.rerun_pyannote and os.path.exists(pya_txt):
            print(f"[fusion] reusing pyannote turns: {pya_txt}")
            pya = read_pipeline_txt(pya_txt)
        else:
            print("[fusion] running pyannote 3.1")
            pya = run_pyannote(wav)
        print(f"[fusion] pyannote: {len(pya)} turns, {len(set(t[2] for t in pya))} speakers")

        print("[fusion] running DiariZen")
        dzn = run_diarizen(wav)
        print(f"[fusion] diarizen: {len(dzn)} turns, {len(set(t[2] for t in dzn))} speakers")

        # DOVER-Lap fusion via CLI (rank-weighted voting)
        uri = "audio"
        r1, r2 = os.path.join(td, "pya.rttm"), os.path.join(td, "dzn.rttm")
        rf = os.path.join(td, "fused.rttm")
        turns_to_rttm(pya, uri, r1)
        turns_to_rttm(dzn, uri, r2)
        dover = os.path.join(os.path.dirname(sys.executable), "dover-lap")
        cmd = [dover, rf, r1, r2]
        w = os.getenv("DOVER_WEIGHTS")
        if w:
            cmd += ["--custom-weight", *w.split(",")]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        fused = rttm_to_turns(rf)

    # normalize speaker names to SPEAKER_xx, ordered by first appearance
    names = {}
    for _, _, spk in fused:
        if spk not in names:
            names[spk] = f"SPEAKER_{len(names):02d}"
    with open(out_txt, "w") as f:
        for start, stop, spk in fused:
            f.write(f"start={start:.3f}s stop={stop:.3f}s speaker_{names[spk]}\n")
    print(f"[fusion] fused: {len(fused)} turns, {len(names)} speakers -> {out_txt}")


if __name__ == "__main__":
    main()

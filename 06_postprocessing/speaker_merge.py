"""
speaker_merge.py - Merge Consecutive Same-Speaker Segments + KIRTAN Gaps

Core post-processing:
  1. Overlaps Whisper segments with PyAnnote diarization to assign speakers
  2. Merges consecutive segments with the same speaker name
  3. Inserts [KIRTAN] markers at 108s+ silence gaps
  4. Enforces 2800-char max per output object (splits on sentence boundaries)

Usage: python speaker_merge.py <lecture_id>
"""
import json, re, sys, os
from pydub import AudioSegment

GAP_THRESH = 108; OVERLAP_MIN = 0.5; MAX_CHARS = 2800
MAIN_SPK = os.getenv("MAIN_SPEAKER_NAME", "Vaisesika Dasa")
GMAP_PATH = os.getenv("GLOBAL_MAP_PATH", "global_map.json")
W_ROOT = os.getenv("WHISPER_OUTPUT_DIR", "FTS_MP3_whisper/")
P_ROOT = os.getenv("PYANNOTE_OUTPUT_DIR", "FTS_MP3_pyannote/")
A_ROOT = os.getenv("AUDIO_ROOT", ".")

def hms(s):
    h, r = divmod(int(s), 3600); m, s = divmod(r, 60); return f"{h:02d}:{m:02d}:{s:02d}"

def parse_dia(line):
    m = re.search(r"start=([\d.]+)s\s+stop=([\d.]+)s\s+speaker_(\w+)", line)
    return {"start": float(m[1]), "end": float(m[2]), "speaker": m[3]} if m else None

def best_speaker(s, e, timeline):
    best, mx = None, 0.0
    for entry in timeline:
        ov = max(0, min(e, entry["end"]) - max(s, entry["start"]))
        if ov > mx: mx, best = ov, entry["speaker"]
    return best if mx >= OVERLAP_MIN else None

def split_text(text, mx):
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= mx: cur += (" " if cur else "") + s
        else:
            if cur: chunks.append(cur.strip())
            if len(s) > mx:
                words = s.split(); tmp = ""
                for w in words:
                    if len(tmp) + len(w) + 1 <= mx: tmp += (" " if tmp else "") + w
                    else:
                        if tmp: chunks.append(tmp.strip())
                        tmp = w
                if tmp: chunks.append(tmp.strip())
            else: cur = s
    if cur: chunks.append(cur.strip())
    return chunks

def process_file(lid, output_dir=None):
    w_path = os.path.join(W_ROOT, lid, "transcript.json")
    p_path = os.path.join(P_ROOT, f"{lid}.txt")
    out_path = os.path.join(output_dir or "v2", f"{lid}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(w_path, encoding="utf-8") as f: wdata = json.load(f)
    dia = []
    if os.path.exists(p_path):
        with open(p_path) as f:
            for line in f:
                p = parse_dia(line.strip())
                if p: dia.append(p)
    gmap = {}
    if os.path.exists(GMAP_PATH):
        with open(GMAP_PATH) as f: gmap = json.load(f)
    smap = gmap.get(lid, {})

    segs = []
    for seg in wdata:
        s, e = seg.get("start", 0.0), seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if not text: continue
        raw = best_speaker(s, e, dia)
        spk = smap.get(raw, MAIN_SPK) if raw else MAIN_SPK
        segs.append({"speaker": spk, "start": s, "end": e, "text": text})

    # Merge consecutive same-speaker
    merged = []
    for seg in segs:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += " " + seg["text"]; merged[-1]["end"] = seg["end"]
        else: merged.append(seg.copy())

    # Build output with KIRTAN gaps and char limits
    output = []
    for i, seg in enumerate(merged):
        if i > 0 and seg["start"] - merged[i-1]["end"] >= GAP_THRESH:
            output.append({MAIN_SPK: {"text": "[KIRTAN]", "start": hms(merged[i-1]["end"]), "end": hms(seg["start"])}})
        text = seg["text"].strip()
        if len(text) <= MAX_CHARS:
            output.append({seg["speaker"]: {"text": text, "start": hms(seg["start"]), "end": hms(seg["end"])}})
        else:
            chunks = split_text(text, MAX_CHARS)
            dur, tc = seg["end"] - seg["start"], sum(len(c) for c in chunks)
            t = seg["start"]
            for ch in chunks:
                d = (len(ch) / max(tc, 1)) * dur
                output.append({seg["speaker"]: {"text": ch, "start": hms(t), "end": hms(t + d)}}); t += d

    with open(out_path, "w", encoding="utf-8") as f: json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {lid}: {len(output)} blocks -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) == 2: process_file(sys.argv[1])
    elif len(sys.argv) >= 5:
        lid = os.path.splitext(os.path.basename(sys.argv[4]))[0]
        process_file(lid, os.path.dirname(sys.argv[3]))
    else: print("Usage: python speaker_merge.py <lecture_id>")

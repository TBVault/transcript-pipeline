"""
correct_gemini_timesteps.py - Bridge Segment Time Interpolation

Interpolates start/end times for BRIDGE_SEGMENTs from neighbors.

Usage: python correct_gemini_timesteps.py <lecture_dir/>
"""
import json, sys, os, re

def load_json(p):
    with open(p) as f: return json.load(f)
def save_json(o, p):
    with open(p, "w", encoding="utf-8") as f: json.dump(o, f, ensure_ascii=False, indent=2)

def interpolate(segments):
    n = len(segments)
    prev_w, next_w = [None]*n, [None]*n
    last = None
    for i in range(n):
        if segments[i].get("type") == "WHISPER_SEGMENT": last = i
        prev_w[i] = last
    last = None
    for i in range(n-1, -1, -1):
        if segments[i].get("type") == "WHISPER_SEGMENT": last = i
        next_w[i] = last
    for i in range(n):
        if segments[i].get("type") != "BRIDGE_SEGMENT": continue
        pi, ni = prev_w[i], next_w[i]
        if pi is not None and ni is not None:
            segments[i]["start"] = segments[pi].get("end", 0.0)
            segments[i]["end"] = segments[ni].get("start", segments[i]["start"])
        elif pi is not None:
            segments[i]["start"] = segments[pi].get("end", 0.0)
            wc = len((segments[i].get("gemini_transcript") or "").split())
            segments[i]["end"] = segments[i]["start"] + wc * 0.3
        elif ni is not None:
            segments[i]["end"] = segments[ni].get("start", 0.0)
            wc = len((segments[i].get("gemini_transcript") or "").split())
            segments[i]["start"] = max(0.0, segments[i]["end"] - wc * 0.3)
    return segments

if __name__ == "__main__":
    d = sys.argv[1]
    inp = os.path.join(d, "segments_with_whisper_and_gemini_filled.json")
    out = os.path.join(d, "segments_corrected.json")
    segs = load_json(inp)
    corrected = interpolate(segs)
    save_json(corrected, out)
    bc = sum(1 for s in corrected if s.get("type") == "BRIDGE_SEGMENT")
    print(f"[DONE] {bc} bridge segments interpolated -> {out}")

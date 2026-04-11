"""
fuzz.py - GPU-Accelerated Needleman-Wunsch Token Alignment

Aligns Whisper word-level tokens against Gemini's full-text transcription.
Maps Gemini's superior text onto Whisper's precise timestamps.

Usage: python fuzz.py <lecture_dir/>
Input:  lecture_dir/segments_with_whisper_and_gemini.json
Output: lecture_dir/segments_with_whisper_and_gemini_filled.json
"""
import json, re, sys
import numpy as np

INPUT_PATH = sys.argv[1] + "/segments_with_whisper_and_gemini.json"
OUTPUT_PATH = sys.argv[1] + "/segments_with_whisper_and_gemini_filled.json"
MATCH, MISMATCH, GAP = 10, -5, -1

with open(INPUT_PATH, "r") as f: items = json.load(f)

# Build Whisper token sequence
whisper_segs, w_tokens, w_map, seg_counts = [], [], [], {}
for item in items:
    if item["type"] == "WHISPER_SEGMENT":
        whisper_segs.append(item)
        idx = len(whisper_segs) - 1
        count = 0
        for t in (item.get("whisper_transcript") or "").split():
            norm = re.sub(r"[^\w]", "", t.lower())
            if norm: w_tokens.append(norm); w_map.append(idx); count += 1
        seg_counts[idx] = count

# Build Gemini token sequence
g_raw, g_norm = [], []
for item in sorted(items, key=lambda x: x["start"]):
    if item["type"] == "WHISPER_GROUP":
        for t in (item.get("gemini_transcript") or "").split():
            norm = re.sub(r"[^\w]", "", t.lower())
            if norm: g_raw.append(t); g_norm.append(norm)

# Integer encode
vocab = list(set(w_tokens + g_norm))
tok2id = {t: i for i, t in enumerate(vocab)}
w_np = np.array([tok2id[t] for t in w_tokens])
g_np = np.array([tok2id[t] for t in g_norm])
n, m = len(w_np), len(g_np)
print(f"Whisper: {n}, Gemini: {m}, Vocab: {len(vocab)}")

# NW alignment
score = np.zeros((n+1, m+1), dtype=np.int32)
for i in range(1, n+1): score[i][0] = i * GAP
for j in range(1, m+1): score[0][j] = j * GAP
for i in range(1, n+1):
    for j in range(1, m+1):
        s = MATCH if w_np[i-1] == g_np[j-1] else MISMATCH
        score[i][j] = max(score[i-1][j-1]+s, score[i-1][j]+GAP, score[i][j-1]+GAP)

# Traceback
alignment = []
i, j = n, m
while i > 0 and j > 0:
    s = MATCH if w_np[i-1] == g_np[j-1] else MISMATCH
    if score[i][j] == score[i-1][j-1]+s: alignment.append((i-1, j-1)); i -= 1; j -= 1
    elif score[i][j] == score[i-1][j]+GAP: i -= 1
    else: j -= 1
alignment.reverse()

# Map Gemini text onto Whisper segments
seg_gemini = {}
for wi, gi in alignment:
    si = w_map[wi]
    seg_gemini.setdefault(si, []).append(g_raw[gi])
for si, toks in seg_gemini.items():
    whisper_segs[si]["gemini_transcript"] = " ".join(toks)
    wc, gc = seg_counts.get(si, 1), len(toks)
    whisper_segs[si]["alignment_score"] = round(min(wc, gc) / max(wc, gc, 1) * 10, 2)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print(f"[DONE] {len(alignment)} pairs aligned -> {OUTPUT_PATH}")

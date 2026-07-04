import json
import re
import sys
import torch

# --- CONFIG ---

INPUT_PATH = sys.argv[1] + "/segments_with_whisper_and_gemini.json"
OUTPUT_PATH = sys.argv[1] + "/segments_with_whisper_and_gemini_filled.json"

# SCORING (Integers for speed)

MATCH_SCORE = 10
MISMATCH_SCORE = -5
GAP_SCORE = -1  # Lower gap penalty encourages "ghost segments" over bad matches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on {device}")


def normalize(s):
    s = s.replace("-", " ")
    return re.sub(r"[^\w\s]", "", s.lower()).split()


def get_tokens(s):
    return s.split()


print(f"Loading {INPUT_PATH}...")

with open(INPUT_PATH, "r") as f:
    items = json.load(f)

# 1. Prepare Data (ROBUST SYNC)

whisper_segs = []
w_tokens = []  # Normalized tokens for alignment
w_map = []  # Index to segment

# (old initial pass cleared immediately; kept structure from your file)
whisper_segs = []
w_tokens = []
w_map = []

seg_token_counts = {}  # seg_idx -> number of whisper tokens

for idx, item in enumerate(items):
    if item["type"] == "WHISPER_SEGMENT":
        text = item.get("whisper_transcript") or ""
        raw_toks = text.split()
        whisper_segs.append(item)
        seg_idx = len(whisper_segs) - 1
        count = 0
        for t in raw_toks:
            norm_t = re.sub(r"[^\w]", "", t.lower())
            if norm_t:
                w_tokens.append(norm_t)
                w_map.append(seg_idx)
                count += 1
        seg_token_counts[seg_idx] = count  # store per-segment token count

gemini_raw = []
gemini_norm = []

sorted_items = sorted(items, key=lambda x: x["start"])

for item in sorted_items:
    if item["type"] == "WHISPER_GROUP":
        text = item.get("gemini_transcript") or ""
        
        #commenting this out because this is important
        #text = text.replace("[MISSING AUDIO - GAP FILLED]", "")
        raw_toks = text.split()
        for t in raw_toks:
            norm_t = re.sub(r"[^\w]", "", t.lower())
            if norm_t:
                gemini_raw.append(t)
                gemini_norm.append(norm_t)

# 2. Encode Tokens to Integers for GPU

vocab = list(set(w_tokens + gemini_norm))
token_to_id = {t: i for i, t in enumerate(vocab)}

w_ids = torch.tensor([token_to_id[t] for t in w_tokens], device=device, dtype=torch.int32)
g_ids = torch.tensor([token_to_id[t] for t in gemini_norm], device=device, dtype=torch.int32)

# 3. Needleman-Wunsch (Numba CPU fallback)

from numba import cuda
import numpy as np


@cuda.jit
def needleman_wunsch_kernel(w_ids, g_ids, score_matrix, dir_matrix, match, mismatch, gap):
    # Placeholder kernel (unused) – kept to match your original structure
    return


from numba import njit


@njit
def compute_score_matrix(w_ids, g_ids, match, mismatch, gap):
    N = len(w_ids)
    M = len(g_ids)
    scores = np.zeros((N + 1, M + 1), dtype=np.int32)
    directions = np.zeros((N + 1, M + 1), dtype=np.int8)

    # Init
    for i in range(1, N + 1):
        scores[i, 0] = i * gap
        directions[i, 0] = 2  # Up

    for j in range(1, M + 1):
        scores[0, j] = j * gap
        directions[0, j] = 3  # Left

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if w_ids[i - 1] == g_ids[j - 1]:
                s_match = scores[i - 1, j - 1] + match
            else:
                s_match = scores[i - 1, j - 1] + mismatch
            s_del = scores[i - 1, j] + gap
            s_ins = scores[i, j - 1] + gap

            if s_match >= s_del and s_match >= s_ins:
                scores[i, j] = s_match
                directions[i, j] = 1
            elif s_del >= s_ins:
                scores[i, j] = s_del
                directions[i, j] = 2
            else:
                scores[i, j] = s_ins
                directions[i, j] = 3

    return directions


print("Compiling & Running Numba (CPU Optimized)...")

w_ids_np = np.array([token_to_id[t] for t in w_tokens], dtype=np.int32)
g_ids_np = np.array([token_to_id[t] for t in gemini_norm], dtype=np.int32)

N = len(w_tokens)
M = len(gemini_norm)

print(f"Matrix Size: {N} x {M}")
dir_matrix = compute_score_matrix(w_ids_np, g_ids_np, MATCH_SCORE, MISMATCH_SCORE, GAP_SCORE)

# 4. Traceback

print("Traceback...")

aligned_w = []
aligned_g = []

i, j = N, M

while i > 0 or j > 0:
    d = dir_matrix[i, j]
    if d == 1:
        aligned_w.append(i - 1)
        aligned_g.append(j - 1)
        i -= 1
        j -= 1
    elif d == 2:
        aligned_w.append(i - 1)
        aligned_g.append(None)
        i -= 1
    elif d == 3:
        aligned_w.append(None)
        aligned_g.append(j - 1)
        j -= 1
    else:  # Boundary
        if i > 0:
            aligned_w.append(i - 1)
            aligned_g.append(None)
            i -= 1
        elif j > 0:
            aligned_w.append(None)
            aligned_g.append(j - 1)
            j -= 1

aligned_w.reverse()
aligned_g.reverse()

print("Calculating segment scores...")

segment_scores = {}

for k in range(len(aligned_w)):
    w_idx = aligned_w[k]
    g_idx = aligned_g[k]

    if w_idx is not None:
        seg_i = w_map[w_idx]
        if seg_i not in segment_scores:
            segment_scores[seg_i] = 0

        if g_idx is not None:
            if w_tokens[w_idx] == gemini_norm[g_idx]:
                segment_scores[seg_i] += MATCH_SCORE
            else:
                segment_scores[seg_i] += MISMATCH_SCORE
        else:
            segment_scores[seg_i] += GAP_SCORE

# NEW: per-segment last aligned whisper index, used to push tails to bridge

last_aligned_w_idx_for_seg = {}  # seg_idx -> max w_idx that has some g_idx

for k in range(len(aligned_w)):
    w_idx = aligned_w[k]
    g_idx = aligned_g[k]
    if w_idx is not None and g_idx is not None:
        seg_idx = w_map[w_idx]
        prev = last_aligned_w_idx_for_seg.get(seg_idx, -1)
        if w_idx > prev:
            last_aligned_w_idx_for_seg[seg_idx] = w_idx


def score_segment_local_raw(whisper_text: str, gemini_text: str) -> int:
    w_raw = (whisper_text or "").split()
    g_raw = (gemini_text or "").split()

    w_norm = [
        re.sub(r"[^\w]", "", t.lower())
        for t in w_raw
        if re.sub(r"[^\w]", "", t.lower())
    ]
    g_norm = [
        re.sub(r"[^\w]", "", t.lower())
        for t in g_raw
        if re.sub(r"[^\w]", "", t.lower())
    ]

    N = len(w_norm)
    M = len(g_norm)

    if N == 0 and M == 0:
        return 0
    if N == 0:
        return M * GAP_SCORE
    if M == 0:
        return N * GAP_SCORE

    scores = [[0] * (M + 1) for _ in range(N + 1)]

    for i in range(1, N + 1):
        scores[i][0] = scores[i - 1][0] + GAP_SCORE
    for j in range(1, M + 1):
        scores[0][j] = scores[0][j - 1] + GAP_SCORE

    for i in range(1, N + 1):
        wi = w_norm[i - 1]
        for j in range(1, M + 1):
            gj = g_norm[j - 1]
            if wi == gj:
                s_match = scores[i - 1][j - 1] + MATCH_SCORE
            else:
                s_match = scores[i - 1][j - 1] + MISMATCH_SCORE
            s_del = scores[i - 1][j] + GAP_SCORE
            s_ins = scores[i][j - 1] + GAP_SCORE
            scores[i][j] = max(s_match, s_del, s_ins)

    raw_score = scores[N][M]
    return raw_score


def score_segment_local(whisper_text: str, gemini_text: str) -> float:
    """
    Normalized local alignment score: raw_score / max(#whisper_tokens, 1).
    """
    w_raw = (whisper_text or "").split()
    w_norm = [
        re.sub(r"[^\w]", "", t.lower())
        for t in w_raw
        if re.sub(r"[^\w]", "", t.lower())
    ]
    N = len(w_norm)
    raw_score = score_segment_local_raw(whisper_text, gemini_text)
    norm_factor = max(N, 1)
    return raw_score / norm_factor


# 5. Reconstruction

print("Reconstructing...")

final_output = []
current_seg_idx = -1
current_gemini_buffer = []

for k in range(len(aligned_w)):
    w_idx = aligned_w[k]
    g_idx = aligned_g[k]

    if w_idx is None:
        if g_idx is not None:
            # bridge text not aligned to any whisper token
            current_gemini_buffer.append(gemini_raw[g_idx])
        continue

    seg_idx = w_map[w_idx]

    if seg_idx != current_seg_idx:
        # Flush ghost / pre-segment Gemini as a bridge
        if current_gemini_buffer:
            t_start = whisper_segs[current_seg_idx]["end"] if current_seg_idx != -1 else 0.0
            t_end = whisper_segs[seg_idx]["start"]
            t_start = min(t_start, t_end)

            final_output.append(
                {
                    "type": "GEMINI_BRIDGE_SEGMENT",
                    "start": t_start,
                    "end": t_end,
                    "gemini_transcript": " ".join(current_gemini_buffer),
                    "primary_type": "LECTURE",
                    "note": "Gap",
                }
            )
            current_gemini_buffer = []

        # New Whisper segment
        out = whisper_segs[seg_idx].copy()
        raw_score = segment_scores.get(seg_idx, 0)
        length = seg_token_counts.get(seg_idx, 0)
        if length > 0:
            norm_score = raw_score / float(length)
        else:
            norm_score = 0.0
        out["alignment_score"] = norm_score
        out["gemini_transcript"] = ""
        final_output.append(out)
        current_seg_idx = seg_idx

    # Decide where Gemini token goes (segment vs bridge tail)
    if g_idx is not None:
        if current_seg_idx != -1:
            last_w_for_seg = last_aligned_w_idx_for_seg.get(current_seg_idx, None)

            # If this whisper index is at or before the last aligned whisper token
            # for the segment, keep Gemini inside; otherwise it is tail → bridge.
            if last_w_for_seg is not None and w_idx <= last_w_for_seg:
                txt = final_output[-1]["gemini_transcript"]
                if txt:
                    txt += " "
                txt += gemini_raw[g_idx]
                final_output[-1]["gemini_transcript"] = txt
            else:
                current_gemini_buffer.append(gemini_raw[g_idx])
        else:
            # No current segment yet: pre-Whisper Gemini
            current_gemini_buffer.append(gemini_raw[g_idx])

# Flush any residual buffer into the last segment's transcript
if current_gemini_buffer and final_output:
    txt = final_output[-1]["gemini_transcript"]
    if txt:
        txt += " "
    txt += " ".join(current_gemini_buffer)
    final_output[-1]["gemini_transcript"] = txt
    current_gemini_buffer = []

# Normalize bridge-like types then merge bridges

merged = []
buffer_bridge = None


def is_bridge(o):
    return o.get("type") in ("GEMINI_BRIDGE_SEGMENT", "WHISPER_BRIDGE_SEGMENT")


# Normalize all bridge-like types to BRIDGE_SEGMENT
for obj in final_output:
    if obj.get("type") in ("GEMINI_BRIDGE_SEGMENT", "WHISPER_BRIDGE_SEGMENT"):
        obj["type"] = "BRIDGE_SEGMENT"

# Single merge pass: collapse any consecutive BRIDGE_SEGMENTs with same primary_type
for obj in final_output:
    if obj.get("type") == "BRIDGE_SEGMENT":
        obj_primary = obj.get("primary_type", "LECTURE")
        if buffer_bridge is None:
            buffer_bridge = {
                "type": "BRIDGE_SEGMENT",
                "start": obj["start"],
                "end": obj["end"],
                "primary_type": obj_primary,
                "gemini_transcript": obj.get("gemini_transcript", "") or "",
                "whisper_transcript": obj.get("whisper_transcript", "") or "",
                "note": (obj.get("note", "") or "").strip() or "Merged bridge",
            }
        else:
            if buffer_bridge.get("primary_type") != obj_primary:
                merged.append(buffer_bridge)
                buffer_bridge = {
                    "type": "BRIDGE_SEGMENT",
                    "start": obj["start"],
                    "end": obj["end"],
                    "primary_type": obj_primary,
                    "gemini_transcript": obj.get("gemini_transcript", "") or "",
                    "whisper_transcript": obj.get("whisper_transcript", "") or "",
                    "note": (obj.get("note", "") or "").strip() or "Merged bridge",
                }
            else:
                buffer_bridge["end"] = obj["end"]
                gt = obj.get("gemini_transcript", "") or ""
                wt = obj.get("whisper_transcript", "") or ""
                if gt:
                    if buffer_bridge["gemini_transcript"]:
                        buffer_bridge["gemini_transcript"] += " "
                    buffer_bridge["gemini_transcript"] += gt
                if wt:
                    if buffer_bridge["whisper_transcript"]:
                        buffer_bridge["whisper_transcript"] += " "
                    buffer_bridge["whisper_transcript"] += wt
                note = (obj.get("note", "") or "").strip()
                if note:
                    if buffer_bridge.get("note"):
                        buffer_bridge["note"] += " " + note
                    else:
                        buffer_bridge["note"] = note
    else:
        if buffer_bridge is not None:
            merged.append(buffer_bridge)
            buffer_bridge = None
        merged.append(obj)

if buffer_bridge is not None:
    merged.append(buffer_bridge)

final_output = merged

# Remove whisper_transcript from all bridge segments
for obj in final_output:
    if obj.get("type") == "BRIDGE_SEGMENT":
        obj.pop("whisper_transcript", None)



def is_whisper(o):
    return o.get("type") == "WHISPER_SEGMENT"

def is_perfect(o):
    return o.get("alignment_score", 0) >= 9.99

def token_count(s: str) -> int:
    return len((s or "").split())

print("Running post-process PASS 2 (Case A/B/C)...")

final_output = final_output
processed_output = []
i = 0
dropped_count = 0
rate_pushed_count = 0
zero_pushed_count = 0

while i < len(final_output):
    obj = final_output[i]

    if obj.get("type") != "BRIDGE_SEGMENT":
        processed_output.append(obj)
        i += 1
        continue

    prev_w_idx = -1
    next_w_idx = -1
    
    # Check neighbors in processed_output (prev) and final_output (next)
    if processed_output and is_whisper(processed_output[-1]):
        prev_w_idx = len(processed_output) - 1
    if i + 1 < len(final_output) and is_whisper(final_output[i+1]):
        next_w_idx = i + 1

    if prev_w_idx != -1 and next_w_idx != -1:
        prev_seg = processed_output[prev_w_idx]
        next_seg = final_output[next_w_idx]

        p_perf = is_perfect(prev_seg)
        n_perf = is_perfect(next_seg)
        duration = obj.get("end", 0) - obj.get("start", 0)
        b_text = obj.get("gemini_transcript", "") or ""

        # --- Case A: Zero-duration + Mixed Perfection -> Push to the non-10.0 one ---
        if abs(duration) <= 0.001 and (p_perf != n_perf):
            # Identify target: the one that is NOT perfect
            if not p_perf:
                target_seg = prev_seg
                target_label = "PREV (imperfect)"
            else:
                target_seg = next_seg
                target_label = "NEXT (imperfect)"

            old_g = target_seg.get("gemini_transcript", "") or ""
            target_seg["gemini_transcript"] = (old_g + " " + b_text).strip() if old_g else b_text
            
            # Recalculate score for the modified segment
            target_seg["alignment_score"] = score_segment_local(
                target_seg.get("whisper_transcript", ""),
                target_seg.get("gemini_transcript", "")
            )
            
            print(f"[ZERO_PUSH] Zero-duration bridge at {obj['start']} pushed to {target_label} segment.")
            zero_pushed_count += 1
            i += 1
            continue

        # --- Case B: both perfect & zero-duration -> delete ---
        if p_perf and n_perf and abs(duration) <= 0.001:
            print(f"[DROP] Zero-duration bridge at {obj['start']} between perfect segments deleted.")
            dropped_count += 1
            i += 1
            continue

        # --- Case C: both perfect, short & high token rate -> push to lighter segment ---
        if p_perf and n_perf and 0 < duration < 2.0:
            g_tokens = b_text.split()
            rate = len(g_tokens) / duration if duration > 0 else 0.0
            
            # Only push if rate is unusually high (implied phantom text)
            if rate > 5.0:
                def seg_rate(seg):
                    w = (seg.get("whisper_transcript", "") or "").split()
                    d = seg.get("end", 0) - seg.get("start", 0)
                    return (len(w) / d) if d > 0 else 0.0

                prev_rate = seg_rate(prev_seg)
                next_rate = seg_rate(next_seg)

                # Push to the one with lower token density (lighter)
                target_seg = prev_seg if prev_rate < next_rate else next_seg
                target_label = "PREV" if target_seg is prev_seg else "NEXT"

                old_g = target_seg.get("gemini_transcript", "") or ""
                target_seg["gemini_transcript"] = (old_g + " " + b_text).strip() if old_g else b_text
                
                target_seg["alignment_score"] = score_segment_local(
                    target_seg.get("whisper_transcript", ""),
                    target_seg.get("gemini_transcript", "")
                )
                
                print(
                    f"[RATE_PUSH] Bridge at {obj['start']} (rate={rate:.2f} t/s, dur={duration:.2f}s) "
                    f"pushed to {target_label} perfect seg "
                    f"(prev_rate={prev_rate:.2f}, next_rate={next_rate:.2f})."
                )
                rate_pushed_count += 1
                i += 1
                continue

        # --- Case D: -1.0 / 10.0 neighbor with bridge -> absorb into the -1.0 seg ---
        if prev_w_idx != -1 and next_w_idx != -1:
            prev_score = prev_seg.get("alignment_score", 0.0)
            next_score = next_seg.get("alignment_score", 0.0)

            # D1: prev = -1.0, next = 10.0  -> absorb into prev
            if abs(prev_score + 1.0) < 1e-3 and is_perfect(next_seg):
                target_seg = prev_seg
                target_label = "PREV(-1.0←bridge, NEXT=10.0)"

            # D2: prev = 10.0, next = -1.0  -> absorb into next
            elif is_perfect(prev_seg) and abs(next_score + 1.0) < 1e-3:
                target_seg = next_seg
                target_label = "NEXT(-1.0←bridge, PREV=10.0)"
            else:
                target_seg = None

            if target_seg is not None:
                old_g = target_seg.get("gemini_transcript", "") or ""
                target_seg["gemini_transcript"] = (old_g + " " + b_text).strip() if old_g else b_text

                target_seg["alignment_score"] = score_segment_local(
                    target_seg.get("whisper_transcript", ""),
                    target_seg.get("gemini_transcript", "")
                )

                print(
                    f"[CASE_D_ABSORB] Bridge at {obj['start']} absorbed into {target_label}."
                )

                i += 1
                continue



    # fallback: keep bridge
    processed_output.append(obj)
    i += 1

final_output = processed_output
print(
    f"PASS 2 done. "
    f"{rate_pushed_count} high-rate perfect-neighbor bridges, "
    f"{zero_pushed_count} zero-len mixed-perfect bridges, "
    f"dropped {dropped_count} zero-len perfect-neighbor bridges."
)

with open(OUTPUT_PATH, "w") as f:
    json.dump(final_output, f, indent=2)
print("Done with bridge push optimization.")


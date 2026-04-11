"""
greedypushing_postprocess.py - Push Orphan Bridge Text Into Neighbors

When a bridge sits between a perfect-score and non-perfect Whisper segment,
its text is appended to the non-perfect neighbor.

Usage: python greedypushing_postprocess.py <input.json> <output.json>
"""
import json, sys

def is_whisper(o): return o.get("type") == "WHISPER_SEGMENT"
def is_perfect(o): return o.get("alignment_score", 0) >= 9.99

def greedy_push(segments):
    output, pushed = [], 0
    i = 0
    while i < len(segments):
        obj = segments[i]
        if obj.get("type") != "BRIDGE_SEGMENT": output.append(obj); i += 1; continue
        prev_w = output[-1] if output and is_whisper(output[-1]) else None
        next_w = segments[i+1] if i+1 < len(segments) and is_whisper(segments[i+1]) else None
        if prev_w and next_w:
            pp, np_ = is_perfect(prev_w), is_perfect(next_w)
            bt = obj.get("gemini_transcript", "") or ""
            target = next_w if pp and not np_ else (prev_w if not pp and np_ else None)
            if target and bt:
                old = target.get("gemini_transcript", "") or ""
                target["gemini_transcript"] = (old + " " + bt).strip() if target is prev_w else (bt + " " + old).strip()
                pushed += 1; i += 1; continue
        output.append(obj); i += 1
    print(f"Greedy push: absorbed {pushed} bridge segments")
    return output

if __name__ == "__main__":
    with open(sys.argv[1]) as f: data = json.load(f)
    result = greedy_push(data)
    with open(sys.argv[2], "w") as f: json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {len(data)} -> {len(result)} segments")

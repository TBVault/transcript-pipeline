"""
merge_transcripts.py - Merge Whisper + Gemini into Alignment Input Format

Combines Whisper word-level segments with Gemini high-quality text segments
into the unified format expected by fuzz.py (Needleman-Wunsch alignment).

Segment types in output:
  WHISPER_SEGMENT  — from Whisper, has whisper_transcript + word-level timing
  WHISPER_GROUP    — Gemini segment overlapping with Whisper coverage
  BRIDGE_SEGMENT   — Gemini segment with no Whisper overlap (timing is estimated)

Usage:
    python merge_transcripts.py <whisper_json> <gemini_json> <output_json>

    whisper_json  Path to Whisper transcript.json
                  Format: [{start, end, text, words:[{word, start, end}]}]
    gemini_json   Path to Gemini transcript.json
                  Format: [{speaker, text, start, end}]
    output_json   Output path for merged segments_with_whisper_and_gemini.json
"""
import json, sys
from pathlib import Path


def overlap(ws, we, gs, ge):
    return max(0.0, min(we, ge) - max(ws, gs))


def merge(whisper_path, gemini_path, output_path):
    with open(whisper_path) as f:
        w_segs = json.load(f)
    with open(gemini_path) as f:
        g_segs = json.load(f)

    # Normalize Gemini: handle both [{speaker, text, start, end}]
    # and {file_id, segments:[{label, start, end, text}]} legacy formats
    if isinstance(g_segs, dict) and "segments" in g_segs:
        g_segs = [
            {"speaker": s.get("label", "LECTURE"), "text": s["text"],
             "start": s["start"], "end": s["end"]}
            for s in g_segs["segments"]
        ]

    items = []

    # Whisper segments → WHISPER_SEGMENT
    for seg in w_segs:
        item = {
            "type": "WHISPER_SEGMENT",
            "start": seg["start"],
            "end": seg["end"],
            "whisper_transcript": seg.get("text", "").strip(),
        }
        if "words" in seg:
            item["words"] = seg["words"]
        items.append(item)

    # Gemini segments → WHISPER_GROUP or BRIDGE_SEGMENT
    for seg in g_segs:
        gs, ge = seg.get("start", 0.0), seg.get("end", 0.0)
        has_overlap = any(
            overlap(ws["start"], ws["end"], gs, ge) > 0.0
            for ws in w_segs
        )
        items.append({
            "type": "WHISPER_GROUP" if has_overlap else "BRIDGE_SEGMENT",
            "start": gs,
            "end": ge,
            "gemini_transcript": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", ""),
        })

    # Sort by start time, break ties by end time
    items.sort(key=lambda x: (x["start"], x["end"]))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    w_count = sum(1 for x in items if x["type"] == "WHISPER_SEGMENT")
    g_count = sum(1 for x in items if x["type"] == "WHISPER_GROUP")
    b_count = sum(1 for x in items if x["type"] == "BRIDGE_SEGMENT")
    print(f"[DONE] {w_count} WHISPER + {g_count} GEMINI + {b_count} BRIDGE → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_transcripts.py <whisper.json> <gemini.json> <output.json>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3])

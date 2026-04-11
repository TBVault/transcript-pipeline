"""
docx_to_json.py - Convert Manual DOCX Transcript to JSON

Usage: python docx_to_json.py <input.docx> <output.json>
"""
import json, re, sys
import docx

def convert(dp, op):
    doc = docx.Document(dp); paras, spk = [], "Unknown"
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t: continue
        if re.match(r"^(Vaisesika Dasa|Audience \d+|Speaker \d+)$", t, re.I):
            spk = t; continue
        paras.append({"speaker": spk, "text": t})
    merged = []
    for p in paras:
        if merged and merged[-1]["speaker"] == p["speaker"]: merged[-1]["text"] += " " + p["text"]
        else: merged.append(p.copy())
    with open(op, "w", encoding="utf-8") as f: json.dump(merged, f, indent=2)
    print(f"Converted {dp} -> {op}")

if __name__ == "__main__":
    if len(sys.argv) < 3: print("Usage: python docx_to_json.py <in.docx> <out.json>"); sys.exit(1)
    convert(sys.argv[1], sys.argv[2])

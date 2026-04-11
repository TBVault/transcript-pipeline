"""
benchmark_json.py - WER + Segmentation F1 Two-Model Comparison

Usage: python benchmark_json.py <gt.json> <model_a.json> <model_b.json> [output.csv]
"""
import json, re, sys
import pandas as pd

SIM_THRESH = 0.6

def norm(t): return [w for w in re.sub(r"[^a-z0-9]", " ", t.lower()).split() if w]

def block_f1(gt, pred):
    g, p = set(norm(gt)), set(norm(pred))
    if not g or not p: return 0.0
    i = len(g & p); pr, rc = i/len(p), i/len(g)
    return 2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else min(d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1)
    return d[len(r)][len(h)] / len(r) if r else 0.0

def load_std(path):
    with open(path, encoding="utf-8") as f: data = json.load(f)
    raw = []
    for item in data:
        if "speaker" in item: raw.append(item)
        else:
            spk = list(item.keys())[0]; v = item[spk]
            raw.append({"speaker": spk, "text": v["text"] if isinstance(v, dict) else str(v)})
    merged = []
    for r in raw:
        if merged and merged[-1]["speaker"] == r["speaker"]: merged[-1]["text"] += " " + r["text"]
        else: merged.append(r.copy())
    return merged

if __name__ == "__main__":
    gt = load_std(sys.argv[1]); pa = load_std(sys.argv[2]); pb = load_std(sys.argv[3])
    out_csv = sys.argv[4] if len(sys.argv) > 4 else "benchmark.csv"
    rows = []; ua, ub = set(), set()
    for i, ref in enumerate(gt):
        def find_best(preds, used):
            best_j, best_s = -1, 0.0
            for j in range(len(preds)):
                if j in used: continue
                s = block_f1(ref["text"], preds[j]["text"])
                if s > best_s: best_s, best_j = s, j
            return best_j, best_s
        ja, sa = find_best(pa, ua); jb, sb = find_best(pb, ub)
        if ja >= 0: ua.add(ja)
        if jb >= 0: ub.add(jb)
        wa = wer(ref["text"], pa[ja]["text"]) if ja >= 0 and sa >= SIM_THRESH else None
        wb = wer(ref["text"], pb[jb]["text"]) if jb >= 0 and sb >= SIM_THRESH else None
        rows.append({"Turn": i, "GT": ref["speaker"], "SegA": round(sa,4), "WER_A": round(wa,4) if wa else "",
                      "SegB": round(sb,4), "WER_B": round(wb,4) if wb else ""})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] {out_csv} ({len(gt)} turns)")

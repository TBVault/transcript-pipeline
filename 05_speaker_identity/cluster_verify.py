"""
cluster_verify.py - Voiceprint-verification speaker identity (replaces the
one-shot global agglomerative clustering in embed_and_cluster_diarized.py).

Why: a single cosine cut over ~44k noisy 8 kbps centroids absorbs by chaining —
the old map assigned MAIN to thousands of centroids that are objectively far
from the main speaker's voice (audience/guest speech attributed to the
teacher). Verification against a robust voiceprint has no chaining pathway.

Algorithm (pure CPU, reads the Stage-B embedding cache):
  1. SEED: lectures whose filename names the main speaker, whose dominant
     pyannote speaker holds >60% of speech and >10 min. Their dominant
     centroids seed the voiceprint.
  2. TRIM: iterate mean -> drop bottom 10% by cosine -> re-mean (4 rounds),
     yielding a voiceprint robust to mistitled/degraded seeds.
  3. VERIFY every (lecture, SPEAKER_xx) centroid:
       cos >= T_MAIN (0.80)  -> MAIN     (clearly him)
       cos <  T_OTHER (0.55) -> AUDIENCE (clearly not him)
       gray zone: adjudicate WITHIN the lecture — compare to the same
       recording's confirmed-MAIN vs confirmed-AUDIENCE centroids (shared
       channel/distortion makes within-lecture cosine far more reliable):
         sim_main - sim_other >= 0.05           -> MAIN
         sim_other > sim_main                   -> AUDIENCE
         no local anchors / inconclusive        -> MAIN only if this speaker
              dominates the lecture (>50% of speech) and cos >= 0.65
  4. NAME: MAIN -> MAIN_SPEAKER_NAME. Others -> "Audience N", numbered
     PER LECTURE by speaking time (the corpus audio cannot support global
     audience identities — see docs/speaker_identity.md sweep — so the old
     map's single global "Audience 1" implied a cross-lecture identity that
     never existed).

Usage:
    python cluster_verify.py <emb_cache_dir> <pyannote_txt_dir> <out_map.json>
Env: MAIN_SPEAKER_NAME (default "Vaisesika Dasa")
     T_MAIN / T_OTHER / SEED_PATTERN to override defaults.
"""
import os, sys, re, json, glob
import numpy as np
from collections import defaultdict

MAIN = os.getenv("MAIN_SPEAKER_NAME", "Vaisesika Dasa")
T_MAIN = float(os.getenv("T_MAIN", "0.80"))
T_OTHER = float(os.getenv("T_OTHER", "0.55"))
T_GRAY_DOM = 0.65          # gray-zone fallback: dominant speaker needs this much
LOCAL_MARGIN = 0.05
SEED_PATTERN = os.getenv("SEED_PATTERN", "aisesika")   # substring of filename
SEED_SHARE, SEED_SEC = 0.6, 600.0
TURN_RE = re.compile(r"start=([\d.]+)s\s+stop=([\d.]+)s\s+speaker_(\w+)")


def load_durations(pyannote_dir):
    dur = {}
    for fn in glob.glob(os.path.join(pyannote_dir, "*.txt")):
        stem = os.path.basename(fn)[:-4]
        by = defaultdict(float)
        for line in open(fn):
            m = TURN_RE.search(line)
            if m and float(m[2]) > float(m[1]):
                spk = m[3] if m[3].startswith("SPEAKER") else "SPEAKER_" + m[3]
                by[spk] += float(m[2]) - float(m[1])
        for spk, s in by.items():
            dur[(stem, spk)] = s
    return dur


def main():
    cache_dir, pyannote_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    dur = load_durations(pyannote_dir)

    keys, X = [], []
    for fn in glob.glob(os.path.join(cache_dir, "*.npy")):
        stem, spk = os.path.basename(fn)[:-4].rsplit("__", 1)
        keys.append((stem, spk))
        X.append(np.load(fn))
    X = np.stack(X).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    sec = np.array([dur.get(k, 0.0) for k in keys])
    lec = defaultdict(list)
    for i, (stem, _) in enumerate(keys):
        lec[stem].append(i)
    print(f"[verify] {len(keys)} centroids, {len(lec)} lectures")

    # 1. seeds
    seeds = []
    for stem, idxs in lec.items():
        if SEED_PATTERN not in stem:
            continue
        tot = sec[idxs].sum()
        if tot <= 0:
            continue
        i = max(idxs, key=lambda i: sec[i])
        if sec[i] / tot > SEED_SHARE and sec[i] > SEED_SEC:
            seeds.append(i)
    if len(seeds) < 20:
        sys.exit(f"[verify] only {len(seeds)} seeds for pattern '{SEED_PATTERN}' — aborting")
    print(f"[verify] {len(seeds)} seed lectures")

    # 2. trimmed voiceprint
    sel = np.array(seeds)
    for _ in range(4):
        vp = X[sel].mean(0)
        vp /= np.linalg.norm(vp)
        cs = X[sel] @ vp
        sel = sel[cs > np.percentile(cs, 10)]
    vp = X[sel].mean(0)
    vp /= np.linalg.norm(vp)
    cos = X @ vp
    print(f"[verify] voiceprint from {len(sel)} trimmed seeds; corpus cos mean={cos.mean():.3f}")

    # 3. verify
    label = np.full(len(keys), -1)          # 1 main, 0 other, -1 undecided
    label[cos >= T_MAIN] = 1
    label[cos < T_OTHER] = 0
    n_gray = int((label == -1).sum())
    stats = defaultdict(int)
    for stem, idxs in lec.items():
        anchors_m = [i for i in idxs if label[i] == 1]
        anchors_o = [i for i in idxs if label[i] == 0]
        tot = sec[idxs].sum()
        for i in idxs:
            if label[i] != -1:
                continue
            sim_m = max((float(X[i] @ X[j]) for j in anchors_m), default=None)
            sim_o = max((float(X[i] @ X[j]) for j in anchors_o), default=None)
            if sim_m is not None and (sim_o is None or sim_m - sim_o >= LOCAL_MARGIN):
                label[i] = 1; stats["gray->main(local)"] += 1
            elif sim_o is not None and (sim_m is None or sim_o > sim_m):
                label[i] = 0; stats["gray->other(local)"] += 1
            elif tot > 0 and sec[i] / tot > 0.5 and cos[i] >= T_GRAY_DOM:
                label[i] = 1; stats["gray->main(dominant)"] += 1
            else:
                label[i] = 0; stats["gray->other(fallback)"] += 1
    print(f"[verify] gray zone: {n_gray} centroids -> {dict(stats)}")

    # 4. name and emit
    gmap = {}
    for stem, idxs in lec.items():
        smap = {}
        others = sorted((i for i in idxs if label[i] == 0), key=lambda i: -sec[i])
        rank = {i: n + 1 for n, i in enumerate(others)}
        for i in idxs:
            smap[keys[i][1]] = MAIN if label[i] == 1 else f"Audience {rank[i]}"
        gmap[stem] = smap
    with open(out_path, "w") as f:
        json.dump(gmap, f, indent=1)

    h_main = sec[label == 1].sum() / 3600
    h_other = sec[label == 0].sum() / 3600
    print(f"[verify] MAIN {h_main:.1f}h ({100*h_main/(h_main+h_other):.1f}%) | "
          f"other {h_other:.1f}h | map -> {out_path}")


if __name__ == "__main__":
    main()

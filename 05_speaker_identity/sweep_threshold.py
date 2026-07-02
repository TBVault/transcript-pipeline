"""
sweep_threshold.py - Pick GLOBAL_THRESH for embed_and_cluster_diarized using
the cached embeddings (no GPU, no re-embedding).

Autonomous criterion: choose the TIGHTEST cosine threshold that still keeps the
main speaker as a single global cluster (fragmenting the main speaker is the
worst error — it would mislabel the teacher's own words as "Audience"), while
separating as many distinct audience voices as possible.

Usage: python sweep_threshold.py <emb_cache_dir> <pyannote_txt_dir>
"""
import os, sys, re
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

TURN_RE = re.compile(r"start=([\d.]+)s\s+stop=([\d.]+)s\s+speaker_(\w+)")
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35]
REF = 0.35  # threshold whose dominant cluster defines the "main speaker" member set


def dur_index(pyannote_dir):
    """{(stem, spk): total_speaking_seconds}."""
    d = {}
    for fn in os.listdir(pyannote_dir):
        if not fn.endswith(".txt"):
            continue
        stem = fn[:-4]
        by = defaultdict(float)
        for line in open(os.path.join(pyannote_dir, fn)):
            m = TURN_RE.search(line)
            if m and float(m[2]) > float(m[1]):
                by[m[3]] += float(m[2]) - float(m[1])
        for spk, sec in by.items():
            d[(stem, spk)] = sec
    return d


def main():
    cache_dir, pyannote_dir = sys.argv[1], sys.argv[2]
    durs = dur_index(pyannote_dir)

    items = []  # (stem, spk, emb, dur)
    for f in os.listdir(cache_dir):
        if not f.endswith(".npy"):
            continue
        emb = np.load(os.path.join(cache_dir, f))
        # filename = "<stem>__<spk>.npy"; spk is the last "__"-field
        base = f[:-4]
        stem, _, spk = base.rpartition("__")
        items.append((stem, spk, emb, durs.get((stem, spk), 0.0)))
    if len(items) < 2:
        print(f"only {len(items)} cached embeddings; need more diarized files first")
        return
    X = np.array([it[2] for it in items])
    print(f"{len(items)} speaker-centroids from {len(set(i[0] for i in items))} lectures\n")

    def cluster(t):
        return AgglomerativeClustering(n_clusters=None, distance_threshold=t,
                                       metric="cosine", linkage="average").fit_predict(X)

    # Define main-speaker member set at the reference (safe) threshold.
    ref_labels = cluster(REF)
    dur_by = defaultdict(float)
    for it, l in zip(items, ref_labels):
        dur_by[l] += it[3]
    ref_main = max(dur_by, key=dur_by.get)
    main_members = {idx for idx, l in enumerate(ref_labels) if l == ref_main}
    print(f"main-speaker member set (@{REF}): {len(main_members)} centroids, "
          f"{dur_by[ref_main]/3600:.1f}h speech\n")

    print(f"{'thr':>5} {'#clusters':>9} {'main_frag':>9} {'aud_ids':>8} {'main_share':>10}")
    best = None
    for t in THRESHOLDS:
        labels = cluster(t)
        n_clusters = len(set(labels))
        # how many distinct clusters do the main members fall into? (1 = intact)
        main_frag = len({labels[i] for i in main_members})
        # main cluster now = the one with most main-members
        cnt = defaultdict(int)
        for i in main_members:
            cnt[labels[i]] += 1
        main_now = max(cnt, key=cnt.get)
        aud_ids = n_clusters - 1
        # main share = fraction of total duration in the main cluster
        dby = defaultdict(float)
        for it, l in zip(items, labels):
            dby[l] += it[3]
        main_share = dby[main_now] / max(sum(dby.values()), 1e-9)
        print(f"{t:>5.2f} {n_clusters:>9} {main_frag:>9} {aud_ids:>8} {main_share:>9.0%}")
        # tightest threshold that keeps main speaker intact (frag==1)
        if main_frag == 1:
            best = t
    print()
    if best is not None:
        print(f"RECOMMEND GLOBAL_THRESH={best} "
              f"(tightest with main speaker intact -> most audience separation, no main fragmentation)")
    else:
        print(f"RECOMMEND GLOBAL_THRESH={REF} (all tested thresholds fragment the main speaker; stay safe)")


if __name__ == "__main__":
    main()

"""
global_clustering.py - Two-Level Cross-Lecture Speaker Clustering

1. LOCAL: Per-lecture agglomerative on WavLM embeddings
2. GLOBAL: Cluster local centroids across lectures for persistent IDs

Usage: python global_clustering.py <embeddings_root/> <output_map.json>
"""
import os, sys, json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from tqdm import tqdm

MAIN_SPEAKER = os.getenv("MAIN_SPEAKER_NAME", "Vaisesika Dasa")
LOCAL_THRESH, GLOBAL_THRESH = 0.3, 0.35

def load_all(root):
    folders = {}
    for entry in os.scandir(root):
        if not entry.is_dir(): continue
        edir = os.path.join(entry.path, "embeddings")
        if not os.path.isdir(edir): continue
        segs = []
        for f in os.listdir(edir):
            if not f.endswith(".npy"): continue
            try:
                e = np.load(os.path.join(edir, f)).flatten()
                if not np.isnan(e).any() and e.size > 0: segs.append({"id": f[:-4], "emb": e})
            except: pass
        if segs: folders[entry.name] = segs
    return folders

def local_cluster(segs, thresh=LOCAL_THRESH):
    if len(segs) < 2:
        for s in segs: s["cluster"] = 0
        return segs
    X = np.array([s["emb"] for s in segs])
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh,
                                      metric="cosine", linkage="average").fit_predict(X)
    for s, l in zip(segs, labels): s["cluster"] = int(l)
    return segs

def main():
    root, out_file = sys.argv[1], sys.argv[2]
    folders = load_all(root)
    print(f"Found {len(folders)} lectures")

    all_centroids = []
    for name, segs in tqdm(folders.items(), desc="Local"):
        segs = local_cluster(segs)
        clusters = {}
        for s in segs: clusters.setdefault(s["cluster"], []).append(s["emb"])
        for c, embs in clusters.items():
            centroid = np.mean(embs, axis=0)
            all_centroids.append({"folder": name, "local": c, "emb": centroid,
                                  "size": len(embs)})
    if not all_centroids: print("No centroids."); return

    X = np.array([c["emb"] for c in all_centroids])
    gl = AgglomerativeClustering(n_clusters=None, distance_threshold=GLOBAL_THRESH,
                                  metric="cosine", linkage="average").fit_predict(X)
    for c, l in zip(all_centroids, gl): c["global"] = int(l)

    sizes = Counter()
    for c in all_centroids: sizes[c["global"]] += c["size"]
    main_cluster = sizes.most_common(1)[0][0]

    gmap = {}
    for c in all_centroids:
        gmap.setdefault(c["folder"], {})[str(c["local"])] = MAIN_SPEAKER if c["global"] == main_cluster else f"Audience {c['global']}"

    with open(out_file, "w") as f: json.dump(gmap, f, indent=2)
    print(f"[DONE] {len(gmap)} lectures -> {out_file}")
    print(f"  Main cluster: {main_cluster} ({sizes[main_cluster]} segs), {len(set(gl))} total clusters")

if __name__ == "__main__":
    if len(sys.argv) < 3: print("Usage: python global_clustering.py <root> <out.json>"); sys.exit(1)
    main()

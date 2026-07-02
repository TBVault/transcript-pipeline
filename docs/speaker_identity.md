# Speaker-identity rollout (Stages A → B → C)

How per-file pyannote diarization becomes persistent cross-corpus speaker
names in the final JSONs, and why the pipeline looks the way it does.
Completed over the tbv corpus on 2026-06-29.

## The three stages

Orchestrated by `jobs/run_speaker_identity.sh` (pass any subset: `A B C`;
default `B C`). All stages are idempotent.

| Stage | Script | Output |
|-------|--------|--------|
| A — diarize | `jobs/diarize_corpus.sh` → `04_diarization/diarization_from_segments_local.py` | `outputs/05_pyannote_diarization/<stem>.txt` (pyannote turns) |
| B — embed + global cluster | `05_speaker_identity/embed_and_cluster_diarized.py` | `outputs/07_speaker_clusters/global_map.json` |
| C — re-merge | `06_postprocessing/speaker_merge.py` over every stem | `outputs/08_final_json/<stem>.json` with real speaker names |

Global clustering is inherently corpus-wide, so this **cannot** be a per-file
stage inside the batch runner: new final JSONs produced by `tbv_batch_01.sh`
get correct labels only because `global_map.json` already exists; a stem
diarized *after* Stage B keeps `MAIN_SPEAKER_NAME` for everything until B/C
are re-run.

## Why `embed_and_cluster_diarized.py` exists

The original pair (`gen_embeddings.py` + `global_clustering.py`) keyed its
output by **time-based agglomerative-cluster indices**, but `speaker_merge.py`
looks up the **pyannote `SPEAKER_xx` label** — the two never lined up, so every
speaker silently collapsed to `MAIN_SPEAKER_NAME`. The replacement closes that
seam: per lecture it embeds up to 60 s of each pyannote speaker's longest turns
(WavLM x-vectors, 30 s window / 15 s step, L2-normalized centroid per
`(lecture, SPEAKER_xx)`), globally clusters the centroids (cosine,
agglomerative), names the largest-by-speaking-time cluster `MAIN_SPEAKER_NAME`
and the rest `Audience N`, and emits `global_map.json` in exactly the shape
`speaker_merge.py` reads: `{"<stem>": {"SPEAKER_00": "<name>", ...}}`.

Embeddings are cached in `outputs/07_speaker_clusters/emb_cache/`
(`EMB_CACHE_DIR`), so re-clustering at a different threshold is near-instant.

## Voiceprint verification (`cluster_verify.py`) — current algorithm (2026-07-02)

The agglomerative map had a systematic failure the threshold sweep couldn't
see: **chain absorption**. Measured against a robust main-speaker voiceprint,
the old map assigned MAIN to 735 centroids with cos < 0.45 (objectively not
him) and ~5,200 more in the 0.45–0.65 gray zone — audience questions and guest
lectures attributed to the teacher in the final JSONs (e.g. "Hare Krishna,
Guru Maharaj, I was just thinking about…" labeled as Vaisesika Dasa).

`cluster_verify.py` replaces the one-shot clustering (it runs as Stage B2 in
`jobs/run_speaker_identity.sh`, after embedding populates the cache):

1. **Seed** from filename priors: lectures titled with the main speaker whose
   dominant pyannote speaker holds >60% of speech and >10 min (685 lectures).
2. **Trimmed voiceprint**: iterate mean → drop bottom decile by cosine → re-mean
   (4 rounds; 448 survivors, seed cos mean 0.984). Robust to mistitled files.
3. **Verify** each (lecture, SPEAKER_xx) centroid: cos ≥ 0.80 → MAIN;
   cos < 0.55 → audience; the ~10.5k gray-zone centroids are adjudicated
   **within their own lecture** by comparing to that recording's confirmed-MAIN
   vs confirmed-audience centroids — same channel and distortion, so local
   cosine separates far better than corpus-global cosine. Fallback for lectures
   with no local anchors: dominant speaker (>50% share) at cos ≥ 0.65 → MAIN.
4. **Name**: audience speakers are numbered per lecture by speaking time.
   NOTE the semantic change: "Audience 1" is now lecture-local. The old global
   "Audience 1" implied one cross-lecture identity, which was false — it was a
   413-hour blob of every audience member, guest, and kirtan singer.

Validation (see git history for the analysis scripts):
- Filename ground truth: dominant == MAIN in 95.1% of 1,747 Vaisesika-titled
  lectures (old map: 96.5%; the 26 differences were inspected and are correct
  demotions — guest swamis teaching, long kirtan chant sections).
- 227.9h of speech moved MAIN → audience; 1.8h audience → MAIN. Text
  spot-checks of moved segments read as disciples' questions, announcements by
  other devotees, and kirtan — not the teacher's lecture voice.
- MAIN share of corpus speech: 87.9% → 81.3%.

Thresholds T_MAIN=0.80 / T_OTHER=0.55 come from the corpus cosine histogram:
verified-him mass sits ≥ 0.82, clearly-not-him lump at 0.28–0.40, smear
between. Env-overridable (T_MAIN, T_OTHER, SEED_PATTERN, MAIN_SPEAKER_NAME).

## GLOBAL_THRESH = 0.35 — superseded (historical)

Decided empirically 2026-06-13 via `05_speaker_identity/sweep_threshold.py`
over cached embeddings from 29 lectures (266 speaker-centroids):

| Threshold | Main-speaker clusters | Audience clusters |
|-----------|----------------------|-------------------|
| 0.35 | 1 | 1 |
| 0.30 | 1 | 1 |
| 0.25 | **3 (fragments)** | — |
| 0.20 | 5 | — |
| 0.15 | 11 | — |

Every threshold that keeps the main speaker intact yields a single audience
cluster: on this 8 kbps audio you **cannot** separate individual audience
members without shattering the main speaker. The deliverable is therefore
binary main-vs-"Audience". 0.35 was chosen over 0.30 for margin from the 0.25
fragmentation cliff. Fragmenting the main speaker is the worst possible error
(the teacher's own words get labeled "Audience"), which is why the sweep
optimizes for main-speaker integrity first.

The sweep's *finding* still stands (no global audience identities are
recoverable), but its *conclusion* — pick one agglomerative threshold — is
superseded by the verification approach above, which avoids both failure
modes at once: the voiceprint can't fragment (there is exactly one MAIN
class), and verification has no chaining pathway for absorption.

## Operational lessons baked into the scripts

- **PyTorch 2.6 `weights_only` monkeypatch**: `torch.load` defaults flipped in
  2.6; without the patch `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")`
  dies with "Weights only load failed" — and did so *silently*, leaving
  `05_pyannote_diarization/` empty while speaker_merge fell back to
  `MAIN_SPEAKER_NAME` for everything. Both `whisper_transcribe.py` and
  `diarization_from_segments_local.py` carry the patch now; any new script that
  loads pyannote needs it too.
- **Diarization is CPU/decode-bound on 8 kbps MP3s**, not GPU-bound. At
  1 worker/GPU the V100s sat at ~0%. `WORKERS_PER_GPU=5` (20 workers on 4
  GPUs) roughly doubled throughput to ~48 files/hr; memory stayed at
  7–14 GB of 32 GB. High load (~90 on 32 cores) was throughput-positive —
  don't cut workers because load looks scary.
- **The diarizer globs the mp3 list once at startup** and, on any decode error
  (e.g. a file mid-rewrite by `recover_empty_mp3s`), writes no `.txt` and never
  retries within that run — so "workers exited" ≠ "corpus diarized".
  Relaunching `jobs/diarize_corpus.sh` is free and self-heals gaps (re-globs,
  skips done stems).
- **The watcher** (`jobs/finish_speaker_identity_when_ready.sh`) exists for
  exactly that reason: it gates on real coverage (`#mp3 − #txt ≤
  DIAR_TOLERANCE`, default 5) rather than "no worker alive", auto-relaunches
  the diarizer to mop up gaps (stall detection, `MAX_RELAUNCHES=4`), then fires
  Stages B and C and logs `[watcher] complete`. Log:
  `outputs/finish_speaker_identity.log`.
  Gotcha: `pkill -f` on the watcher's name matches the shell doing the
  killing — kill it by PID.

## Corpus result (2026-06-29)

4,049/4,051 mp3s diarized; 4,017 lectures embedded → 44,442 speaker-centroids
→ **16 global identities** (main = Vaisesika Dasa at ~10.8 M speaking-seconds,
plus 15 audience identities); 2,693 final JSONs re-merged with real labels.

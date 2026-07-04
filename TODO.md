# Future work

## Fusion-algorithm improvements (parked 2026-07-03 — do AFTER model finalization)

Yardstick incoming: the user is preparing a **32-transcript human benchmark**
and will share it — evaluate every change below against it (replaces the old
"buy ground truth" suggestion; model-vs-model agreement conflates style with
error).

1. **Time-banded NW alignment** — the DP is currently time-blind over the whole
   file; on repetitive text (kirtan: "Hare Kṛṣṇa" x500) it can match phrases
   30 min apart. Both sides carry timestamps: constrain the DP to a band around
   the time-predicted diagonal (±2 min drift). Fixes chant-section pathologies
   AND cuts O(n·m) to O(n·band) — a 10k x 10k lecture drops from a 400 MB
   matrix to a sliver (matters at 41k files x 16 workers).
2. **Diacritic-aware token equivalence** — normalizer keeps Unicode letters, so
   `kṛṣṇa` != `krishna`, `rādhārāṇī` != `radharani`: models render Sanskrit
   differently and these score as MISMATCHES in the highest-value vocabulary.
   Transliteration fold (or partial credit for high char-similarity) — ~20 lines.
3. **Use whisperx word confidences** (currently discarded) — (a) weight
   gap/mismatch penalties so low-confidence whisper words lose to Gemini
   cheaply; (b) emit a per-segment "shaky" flag when alignment score and
   whisper confidence are both low → honest quality metadata for the frontend.
4. **Anchor-split alignment** — pin unique n-grams occurring exactly once in
   both transcripts as anchors, align spans between them independently:
   parallel within a lecture, drift-immune, composes with banding.

Explicitly NOT worth touching per evals: Case A–D bridge heuristics and the
greedy push (ugly but no attributable failure mode).

- **TB-scale fleet prep** (survey 2026-07-03, all passwordless-ssh, no
  permission needed): iGpu21 (2× V100-32G, vdabase OK) ready now; iGpu4
  (TITAN X + 56 cores, vdabase OK) and iGpu5 (TITAN X + 48 cores — NFS server,
  use gently) ready for CPU/API stages; iGpu23 (2080 Ti, untested env);
  iGpu (RTX 3090 Ti — fastest single GPU, but its vdabase lacks whisperx:
  install needed; envs pointer /lab/kiran/envs/iGpu.txt created); iGpu7
  broken NVIDIA driver; iGpu25 up per status page but refuses ssh (needs
  key deployed). Per-machine vdabase envs have DRIFTED — audit/align before
  fanning out the TB run. Storage: /home2 on iGpu4 is kiran-owned (7T-class
  local disks exist per machine); /home2 on iGpu15 is root-owned.

- **Banded Needleman-Wunsch in `03_timestamp_alignment/fuzz.py`** — the current
  full O(n·m) score matrix is a pure-Python/numpy loop (CPU-bound despite the
  "GPU-accelerated" docstring). Whisper/Gemini transcripts of the same audio
  never drift far, so a banded DP (cap |i−j| at a few hundred tokens) gives
  ~50× speedup with no quality loss. Worth doing before any future corpus-wide
  re-alignment pass.
- **17 final JSONs have no global_map entry** (stems missing from the embedding
  cache — diarization or embedding failed for them). Their speakers all fall
  back to MAIN_SPEAKER_NAME. Fix: run the Stage B embedder for just those
  stems (cache makes it incremental), regenerate the map with
  `cluster_verify.py`, re-merge those 17.
- **Guest-speaker naming**: `cluster_verify.py` correctly separates guest
  lecturers from the main speaker, but they get "Audience N" labels. A
  non-main speaker dominating a lecture (>40% share, >20 min) is a guest, not
  audience — could be labeled "Guest Speaker" (or matched against per-guest
  voiceprints seeded from filenames: Jayapataka, Giriraja, Radhanath, …).
- **DiarizEN WavLM-Conformer pass** (~12.7% DER vs pyannote's ~16%) for a
  higher-accuracy diarization re-run on lectures that matter most, if turn
  boundaries ever become the quality bottleneck.

# Future work

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

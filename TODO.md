# Future work

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

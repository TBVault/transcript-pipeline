# Corpus batch operations (`jobs/tbv_batch_01.sh`)

How the full run over `/lab/kiran/tbv_mp3` (4,051 flat MP3s, ~8 kbps mono) is
orchestrated, how failures are tracked, and how to recover each failure class.

## Orchestrator design

`jobs/tbv_batch_01.sh` runs the whole per-file pipeline
(gemini → whisper → merge → fuzz → speaker_merge → `outputs/08_final_json/<stem>.json`)
in repeated passes until nothing is pending:

- **Idempotent**: every stage validates its output JSON and skips if already
  valid. Relaunching the script is always safe and is the standard way to
  resume after a crash or fill coverage gaps.
- **Two lanes per pass**: a single background *Gemini lane* transcribes files
  missing from the cache (writes to `/lab/kiran/gemini_3.0_flash/<stem>/`),
  while 4 *GPU workers* (one per V100) run whisper + alignment + merge. Files
  whose Gemini transcript isn't cached yet log `[WAIT][gemini]` and get picked
  up on a later pass.
- **Download-aware**: files with an open writer (`fuser`) or mtime < 90 s are
  skipped as still-downloading; the loop keeps passing until the wget is gone
  and nothing is pending.
- **Never dies on one file**: a per-file failure writes a marker and moves on.

### Env knobs

| Var | Effect |
|-----|--------|
| `GOOGLE_API_KEY` | Required for the Gemini gap-fill lane |
| `TBV_NO_GEMINI=1` | Never call the API — process only files already in the cache; uncached files are **silently skipped** (the pass reports "0 pending" even though uncached work exists) |
| `TBV_LIMIT=N` | Process at most N files in one pass, then exit (test mode) |

### Launching

Always detach fully, or the run dies with the SSH session (this happened once):

```bash
setsid nohup bash jobs/tbv_batch_01.sh >> outputs/tbv_batch_01.txt 2>&1 < /dev/null &
```

Per-file logs land in `outputs/tbv_logs/<stem>.<stage>.log`.

## Failure markers (`outputs/tbv_failed/`)

Each failed file leaves `outputs/tbv_failed/<stem>.<stage>`. The batch skips
any stem with a marker; **delete the marker to retry** on the next pass.

| Marker | Meaning | Disposition |
|--------|---------|-------------|
| `.nolecture` | Gemini transcript has zero LECTURE segments with text (all NOISE/KIRTAN) | Not a failure — verified genuinely non-lecture audio. Leave alone. |
| `.gemini` | Gemini transcription failed | Was a code bug (see below), now fixed; delete marker + rerun, or use `jobs/recover_gemini_failures.sh` |
| `.whisper` | WhisperX produced no valid transcript | Usually VAD "No active speech" on 8 kbps audio; recover via `jobs/recover_whisper_failures.py` |
| `.merge` / `.fuzz` | Alignment-stage crash | Check `outputs/tbv_logs/<stem>.align.log` |
| `.final` | speaker_merge produced no valid JSON | Check the same log |
| `.convert` | Cached Gemini JSON unparseable | Inspect the cache file |

## Recovery playbook

- **Whisper failures with a cached Gemini transcript** —
  `jobs/recover_whisper_failures.py` (dry-run by default, `--apply` to write).
  Root cause: whisperx's pyannote VAD returns "No active speech found" on the
  ultra-low-bitrate MP3s → 0 segments → the whisper gate discards a file that
  has a perfectly good Gemini transcript. Whisper only refines timestamps;
  the recovery builds the final JSON straight from the Gemini cache, exactly
  as the fuzz fallback would. Recovered 177 of 178 such failures (2026-06-13).
- **Gemini `CRITICAL ERROR: 'end'` failures** — root cause was a code bug, not
  the API: `stitch_segments` in `02_transcription/gemini_transcribe.py` raised
  an uncaught `KeyError` on one malformed segment missing `"end"`, discarding
  the whole file. Fixed (KeyError added to the except → skip the bad segment).
  `jobs/recover_gemini_failures.sh` re-runs the affected stems and clears
  markers.
- **Empty/truncated MP3s** — `jobs/recover_empty_mp3s.sh` re-downloads them.
  Note the interaction with diarization below.

## Gemini transcript cache

`/lab/kiran/gemini_3.0_flash/<stem>/<stem>.json`. The gap-fill lane writes new
transcriptions there too — **never re-transcribe anything already in that
directory**; the cache is the source of truth and API calls cost money.

## Run history / corpus accounting

- 2026-06-09: first run (plain nohup — died silently with the session).
- 2026-06-12: restarted with `TBV_NO_GEMINI=1` (user directive: no API calls).
- 2026-06-14: no-gemini passes exhausted — 2,877 final JSONs, 151 markers.
  Remaining ~1,023 files had no cached transcript and were skipped by design.
- 2026-06-14 ordering directive: finish the **entire speaker-identity
  pipeline first** (see `docs/speaker_identity.md`), then transcribe the
  uncached files.
- 2026-06-29: speaker-identity watcher logged complete (16 global identities,
  2,693 final JSONs re-merged with real speaker labels).
- 2026-07-01: batch briefly relaunched with the API enabled, then **killed
  during its startup scan (zero API calls made)** when the user issued a
  standing directive: *no more Google API transcription, ever* — the 3,055-stem
  cache is final. Marker census at this point: 145 `.nolecture` (legit
  non-lecture) + 3 `.whisper` (unrecoverable, no cached content).
- 2026-07-01: `jobs/whisper_only_finish.sh` launched to finish the remaining
  1,025 uncached files Whisper-only (see below), chaining Stage B + C on
  completion.

## Whisper-only finisher (`jobs/whisper_only_finish.sh`)

Finishes files that have **no** cached Gemini transcript without touching the
API. Per file: `whisper_transcribe` → strip segments to `{start,end,text}` →
write `outputs/04_fuzz_merged/<stem>/transcript.json` (the exact slot
`speaker_merge` and Stage C read) → `speaker_merge` with the existing pyannote
diarization and WavLM `global_map.json` → final JSON. Then it chains
`jobs/run_speaker_identity.sh B C` so the global speaker map and every final
JSON are refreshed corpus-wide.

- Transcript quality on these files is Whisper-grade (no Gemini fusion); each
  such fuzz dir carries a `WHISPER_ONLY` marker file for provenance.
- `WORKERS_PER_GPU` (default 2) × `NUM_GPUS` (4) parallel whisper workers;
  `TBV_LIMIT=N` smoke-test mode skips the B/C chain.
- Same fail-marker conventions as the main batch. Expect some `.whisper`
  markers: whisperx's VAD genuinely finds no speech on some 8 kbps files, and
  here there is no Gemini fallback — those files simply have no transcript.
- Defensive check: a pending stem that *does* have a valid Gemini cache is
  logged `[UNEXPECTED-CACHED]` and skipped rather than silently downgraded to
  Whisper-only.

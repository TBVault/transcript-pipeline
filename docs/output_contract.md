# Current output contract (what the pipeline emits today)

Written for the frontend/ingest integration (see the app-side brief,
`docs/pipeline-integration.md` in the app repo — not yet copied here).
This is the as-is state on 2026-07-03; the "Deltas" section maps it against
the app's requirements as relayed so far.

## Per-lecture final JSON

Path: `outputs/08_final_json/<stem>.json`, where `<stem>` is the mp3 basename
without extension (single flat namespace; current corpus `/lab/kiran/tbv_mp3`,
one main speaker).

Schema — an ordered array of single-key objects:

```json
[
  {"Vaisesika Dasa": {"text": "So today we discuss...", "start": "00:02:15", "end": "00:05:42"}},
  {"Audience 1":     {"text": "Maharaj, can you explain...", "start": "00:05:43", "end": "00:06:01"}},
  {"Vaisesika Dasa": {"text": "[KIRTAN]", "start": "00:06:02", "end": "00:09:30"}}
]
```

- Key = display name, not a slug. Main speaker from `MAIN_SPEAKER_NAME`;
  others are `Audience N`, numbered per lecture by speaking time.
- Timestamps are `HH:MM:SS` strings (not MM:SS).
- Segments are capped at 2800 chars, split on sentence boundaries.
- Gaps >108 s are emitted as `[KIRTAN]` markers.
- No top-level metadata: no id, duration, title, category, or language fields.

## Provenance / sidecars

- `outputs/04_fuzz_merged/<stem>/WHISPER_ONLY` marker = transcript is
  Whisper-only (no Gemini fusion); 1,017 of 3,897 current files.
- `outputs/tbv_failed/<stem>.<stage>` = file intentionally has no final JSON
  (`.nolecture` = kirtan/noise only, `.whisper` = no detectable speech).
- No manifest.jsonl yet.

## Deltas vs the app brief (as relayed 2026-07-03)

| App expects | Pipeline today | Cheapest fix |
|---|---|---|
| `transcripts/<speaker-slug>/<basename>.json` | flat `outputs/08_final_json/` | exporter stage (trivial) |
| stable id = relative audio path | stem only | exporter (trivial, needs input root convention) |
| integer-minute duration | absent | exporter derives from last segment end / ffprobe |
| `MM:SS` section timestamps | `HH:MM:SS` | exporter reformat (lossy >59 min — need app rule) |
| 22 valid speaker slugs | 1 display name + `Audience N` | need the slug list + per-corpus speaker mapping |
| IAST diacritics, category vocab, cleaned titles | none of these | needs the brief's vocab; titles derivable from filenames |
| `pipeline` key for extra metadata | n/a | exporter adds (whisper-only flag, speaker-map version, DER source) |
| `manifest.jsonl` for delta ingest | absent | easy — exporter appends on each new/updated file |
| incremental/idempotent over ~41k growing files | batch is idempotent per-stem | already true; needs pointing at the new tree |

## Open questions for the app side

1. Get us the actual `docs/pipeline-integration.md` (copy it into this repo or
   paste it) — the field names/rules from `server/ingest.js` matter verbatim.
2. The audio NFS (`192.168.1.83:/srv/vani/lectures`) is NOT mounted on iGpu15
   and we have no sudo — mounting needs an admin, or rsync access instead.
3. ~41k files across 22 speakers ≈ 10× the current corpus. Transcription
   policy: Google API is retired for this pipeline (standing directive), so
   new files are Whisper-only unless that changes. GPU-time estimate needed
   once we can see per-speaker file counts/durations.
4. Multi-speaker corpora need per-speaker voiceprint seeding for the identity
   stage (current seeding assumes one main speaker per corpus; the
   filename-prior approach generalizes if filenames name the speaker).

# Bhakti Vault — Lecture Transcription & Diarization Pipeline

End-to-end pipeline for transcribing, diarizing, and indexing ~6,000 hours
of Vaisesika Dasa's lecture recordings. Fuses Gemini's text quality with
Whisper's timestamp precision, then layers PyAnnote + WavLM speaker identity.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Raw MP3 Lectures                      │
│                    (~6,000 hours, decades of recordings)             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  01_preprocessing   │
                    │  • MP3 header fix   │
                    │  • 16kHz resample   │
                    │  • Kirtan detection │
                    └──────┬───────┬──────┘
                           │       │
          ┌────────────────▼┐     ┌▼────────────────────┐
          │ TRANSCRIPTION   │     │ DIARIZATION TRACK   │
          │ TRACK           │     │                     │
          │                 │     │  04_diarization     │
          │ 02_transcription│     │  • PyAnnote 3.1     │
          │ • Gemini Flash  │     │  • DiarizEN WavLM   │
          │   (54s chunks)  │     │    (optional)       │
          │ • WhisperX      │     │                     │
          │   (word-level)  │     │  05_speaker_identity│
          │                 │     │  • WavLM embeddings │
          │ 03_alignment    │     │  • Local clustering │
          │ • NW token fuse │     │  • Global clustering│
          │ • Bridge interp │     │    (cross-lecture)   │
          │ • Greedy push   │     │                     │
          └────────┬────────┘     └──────────┬──────────┘
                   │                         │
                   └────────────┬────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  06_postprocessing    │
                    │  • speaker_merge.py   │
                    │    - Overlap assign   │
                    │    - Same-speaker     │
                    │      merge            │
                    │    - KIRTAN gaps      │
                    │    - 2800-char split  │
                    │  • batch_parallel.py  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  OUTPUT: Structured   │
                    │  JSON per lecture     │
                    │  [{speaker: {text,    │
                    │    start, end}}]      │
                    └──────────────────────┘
```

## Pipeline Stages

### Stage 1: Preprocessing (`01_preprocessing/`)

| Script | Purpose |
|--------|---------|
| `check_mp3_headers.py` | Detect & fix corrupted MP3 headers via ffmpeg re-encode |
| `convert_to_16khz.py` | Batch resample to 16kHz mono WAV for model input |
| `detect_kirtans.py` | Classify segments as LECTURE vs KIRTAN using energy + spectral features |

### Stage 2: Transcription (`02_transcription/`)

| Script | Purpose |
|--------|---------|
| `gemini_transcribe.py` | Send 54s chunks to Gemini Flash for high-quality text + speaker turns |
| `whisper_transcribe.py` | Run WhisperX for word-level timestamps with forced alignment |

**Why two models?** Gemini produces superior text (Sanskrit terms, proper nouns, context)
but has timestamp drift. Whisper has precise word-level timing but worse text quality.
The alignment stage fuses the best of both.

### Stage 3: Timestamp Alignment (`03_timestamp_alignment/`)

| Script | Purpose |
|--------|---------|
| `fuzz.py` | GPU-accelerated Needleman-Wunsch token alignment (Gemini text ← Whisper times) |
| `correct_gemini_timesteps.py` | Interpolate times for BRIDGE_SEGMENTs from neighboring Whisper segments |
| `greedypushing_postprocess.py` | Push orphan bridge text into adjacent non-perfect Whisper segments |

### Stage 4: Diarization (`04_diarization/`)

| Script | Purpose |
|--------|---------|
| `split_audio_dia.py` | PyAnnote 3.1 diarization (single file, timeout + embedding extraction) |
| `diarization_from_segments_local.py` | Multi-GPU batch PyAnnote diarization |
| `diarization_from_segments_cloud.py` | Gemini-assisted diarization (experimental) |
| `diar.py` | Minimal single-file PyAnnote runner for debugging |

### Stage 5: Speaker Identity (`05_speaker_identity/`)

| Script | Purpose |
|--------|---------|
| `gen_embeddings.py` | Extract WavLM x-vector embeddings (30s sliding window, L2-normalized) |
| `global_clustering.py` | Two-level agglomerative clustering: local → global cross-lecture |

### Stage 6: Post-Processing (`06_postprocessing/`)

| Script | Purpose |
|--------|---------|
| `speaker_merge.py` | Assign speakers via overlap, merge consecutive same-speaker, KIRTAN gaps, 2800-char split |
| `batch_parallel.py` | Run speaker_merge across all lectures in parallel |

### Stage 7: Evaluation (`07_evaluation/`)

| Script | Purpose |
|--------|---------|
| `benchmark_json.py` | WER + Segmentation F1 two-model comparison |
| `generate_gt.py` | Extract ground truth from Gemini LECTURE segments |
| `docx_to_json.py` | Convert manual DOCX transcripts to JSON for eval |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/bhakti-vault-pipeline.git
cd bhakti-vault-pipeline
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run single lecture
source .env
./orchestration/run_pipeline.sh <lecture_id>

# 4. Run batch (multi-GPU)
./orchestration/run_batch_gpu.sh v2/
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Generative AI API key for Gemini |
| `HF_TOKEN` | Yes | HuggingFace token for gated PyAnnote models |
| `AUDIO_ROOT` | Yes | Path to MP3 files directory |
| `WHISPER_OUTPUT_DIR` | No | Default: `FTS_MP3_whisper/` |
| `PYANNOTE_OUTPUT_DIR` | No | Default: `FTS_MP3_pyannote/` |
| `VOICEPRINTS_DIR` | No | Default: `FTS_MP3_voiceprints/` |
| `MAIN_SPEAKER_NAME` | No | Default: `Vaisesika Dasa` |

## Key Design Decisions

- **Gemini + Whisper fusion**: Gemini handles Sanskrit/Bengali and context; Whisper provides sub-second timestamps. NW alignment maps one onto the other.
- **Two diarization systems**: PyAnnote 3.1 locally (~16% DER) for throughput. DiarizEN WavLM-Conformer (~12.7% DER) for higher-accuracy passes.
- **Two-level clustering**: Local per-lecture → global cross-lecture for persistent speaker identities.
- **108s KIRTAN threshold**: Gaps >108s are kirtan/bhajan interludes, explicitly marked.
- **2800-char cap**: Downstream search index limit. Long segments split on sentence boundaries.

## Output Format

```json
[
  {"Vaisesika Dasa": {"text": "So today we discuss from Srimad Bhagavatam...", "start": "00:02:15", "end": "00:05:42"}},
  {"Audience 1": {"text": "Maharaj, can you explain...", "start": "00:05:43", "end": "00:06:01"}},
  {"Vaisesika Dasa": {"text": "[KIRTAN]", "start": "00:06:02", "end": "00:09:30"}}
]
```

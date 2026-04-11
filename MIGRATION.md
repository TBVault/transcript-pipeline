# Migration Guide

## Migration Script

```bash
#!/usr/bin/env bash
SRC="./old_flat_scripts"
DST="./bhakti-vault-pipeline"

cp "$SRC/check_mp3_headers.py"       "$DST/01_preprocessing/"
cp "$SRC/convert_to_16khz.py"        "$DST/01_preprocessing/"
cp "$SRC/detect_kirtans.py"          "$DST/01_preprocessing/"
cp "$SRC/gemini_transcribe.py"       "$DST/02_transcription/"
cp "$SRC/whisper_transcribe.py"      "$DST/02_transcription/"
cp "$SRC/fuzz.py"                    "$DST/03_timestamp_alignment/"
cp "$SRC/correct_gemini_timesteps.py" "$DST/03_timestamp_alignment/"
cp "$SRC/greedypushing_postprocess.py" "$DST/03_timestamp_alignment/"
cp "$SRC/split_audio_dia.py"         "$DST/04_diarization/"
cp "$SRC/diarization_from_segments_local.py" "$DST/04_diarization/"
cp "$SRC/diarization_from_segments_cloud.py" "$DST/04_diarization/"
cp "$SRC/diar.py"                    "$DST/04_diarization/"
cp "$SRC/gen_embeddings.py"          "$DST/05_speaker_identity/"
cp "$SRC/global_clustering.py"       "$DST/05_speaker_identity/"
cp "$SRC/speaker_merge.py"           "$DST/06_postprocessing/"
cp "$SRC/batch_parallel.py"          "$DST/06_postprocessing/"
cp "$SRC/benchmark_json.py"          "$DST/07_evaluation/"
cp "$SRC/generate_gt.py"             "$DST/07_evaluation/"
cp "$SRC/docx_to_json.py"            "$DST/07_evaluation/"
```

## Post-Migration Checklist

- [ ] Replace hardcoded `/dev/shm/FTS_MP3_Files/...` with `os.getenv("AUDIO_ROOT")`
- [ ] Replace hardcoded API keys with `os.getenv()` (see SECURITY.md)
- [ ] Update shell script paths in `orchestration/`
- [ ] `pip install -r requirements.txt` in fresh venv
- [ ] Test: `./orchestration/run_pipeline.sh <test_id>`

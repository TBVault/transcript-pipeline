#!/usr/bin/env python3
"""
recover_whisper_failures.py — Recover tbv_batch files that failed at the Whisper
stage but already have a valid Gemini transcript in the cache.

Root cause: whisperx's pyannote VAD returns "No active speech found" on
ultra-low-bitrate (~8 kbps) MP3s, yielding 0 segments. process_stem() then
fails the file at the Whisper gate and discards it — even when Gemini has a
perfectly good lecture transcript. Whisper exists only to refine timestamps via
word alignment; when it produces nothing, the pipeline's own fuzz fallback
already uses Gemini timestamps directly. This script does the same, building the
final JSON straight from the Gemini cache.

For each outputs/tbv_failed/<stem>.whisper marker:
  - load the Gemini cache JSON, keep LECTURE segments with text
  - if any, write outputs/08_final_json/<stem>.json in the final schema
    [ {"Vaisesika Dasa": {"text", "start", "end"}}, ... ]
  - remove the .whisper marker so the batch treats the file as done

Idempotent: never clobbers an existing final JSON. --apply to write; default dry-run.
"""
import os, sys, json, glob

REPO = "/lab/kiran/transcript-pipeline"
GBAK = "/lab/kiran/gemini_3.0_flash"
FAILDIR = os.path.join(REPO, "outputs/tbv_failed")
FINALDIR = os.path.join(REPO, "outputs/08_final_json")
SPEAKER = "Vaisesika Dasa"  # matches MAIN_SPEAKER_NAME default; diarization is disabled

apply = "--apply" in sys.argv


def lecture_segments(gjson):
    with open(gjson) as f:
        data = json.load(f)
    segs = data["segments"] if isinstance(data, dict) and "segments" in data else data
    return [s for s in segs
            if s.get("label", "LECTURE") == "LECTURE" and s.get("text", "").strip()]


recovered = skipped_have_final = skipped_no_content = 0
for marker in sorted(glob.glob(os.path.join(FAILDIR, "*.whisper"))):
    stem = os.path.basename(marker)[:-len(".whisper")]
    final = os.path.join(FINALDIR, stem + ".json")
    if os.path.exists(final) and os.path.getsize(final) > 0:
        skipped_have_final += 1
        if apply:
            os.remove(marker)
        continue
    gjson = os.path.join(GBAK, stem, stem + ".json")
    if not os.path.exists(gjson):
        skipped_no_content += 1
        continue
    try:
        lec = lecture_segments(gjson)
    except Exception as e:
        print(f"[SKIP] {stem}: gemini parse error: {e}")
        skipped_no_content += 1
        continue
    if not lec:
        skipped_no_content += 1
        continue
    out = [{SPEAKER: {"text": s["text"].strip(),
                      "start": s["start"], "end": s["end"]}} for s in lec]
    print(f"[RECOVER] {stem}: {len(out)} segments")
    if apply:
        with open(final, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.remove(marker)
    recovered += 1

mode = "APPLIED" if apply else "DRY-RUN (pass --apply to write)"
print(f"\n=== {mode} ===")
print(f"recoverable (gemini-only final): {recovered}")
print(f"skipped, already had final:      {skipped_have_final}")
print(f"skipped, no gemini content:      {skipped_no_content}")

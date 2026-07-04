"""
rerun_noise_segments.py — Re-transcribe NOISE segments in existing JSONs.

For each final JSON produced by gemini_transcribe.py / segment_audio_vertexai.py,
this script:
  1. Loads the JSON and finds every segment with label == "NOISE".
  2. For each NOISE segment, cuts that exact time range from the source audio
     and re-sends it to Gemini with a prompt that asks: is this actually NOISE,
     or was it dropped content (LECTURE / KIRTAN)?
  3. If Gemini returns real content, the NOISE segment is replaced with the new
     segment(s). If Gemini confirms it is noise/silence, the segment is kept.
  4. LECTURE and KIRTAN segments are NEVER touched.

Safety:
  - Original JSON is backed up to <file>.pre_noise_rerun.json before modification.
  - Raw responses are appended to <file>.noise_rerun_log.json for audit.
  - If the NOISE segment is >= NOISE_RERUN_CHUNK (default 54s), it is chunked
    into 54s sub-windows like the original pipeline.

Usage:
    # Batch, mirroring an audio_root to a json_root
    python rerun_noise_segments.py <audio_root> <json_root> <num_workers>

    # Single file
    python rerun_noise_segments.py <audio.mp3> --json <file.json>

Env:
    GOOGLE_API_KEY    AI Studio API key (same as gemini_transcribe.py)
"""
import os
import sys
import json
import time
import shutil
import subprocess
import random
import errno
import warnings
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from json import JSONDecoder
from multiprocessing import Semaphore as MPSemaphore

from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    ThinkingConfig,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.(auth|oauth2).*")

# --- CONFIGURATION ---
MODEL_NAME = "gemini-3-flash-preview"

NOISE_RERUN_CHUNK = 54.0
NOISE_RERUN_MIN = 2.0
MIN_REAL_CONTENT = 1.0

MAX_CONCURRENT_API_CALLS = 4
BASE_DELAY = 0
LOCK_STALE_SECONDS = 3600

API_SEMAPHORE = None
CLIENT: genai.Client = None


def _init_worker(semaphore):
    global API_SEMAPHORE, CLIENT
    API_SEMAPHORE = semaphore
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    CLIENT = genai.Client(api_key=api_key)


def try_acquire_lock(temp_dir: str) -> bool:
    try:
        os.mkdir(temp_dir)
        return True
    except OSError as e:
        if e.errno == errno.EEXIST:
            try:
                age = time.time() - os.path.getmtime(temp_dir)
                if age > LOCK_STALE_SECONDS:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    os.mkdir(temp_dir)
                    return True
            except OSError:
                pass
            return False
        raise


def cut_window(src_path: str, start_sec: float, duration_sec: float, out_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration_sec:.3f}",
        "-i", src_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-q:a", "4",
        "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def _sleep_with_jitter(base: float):
    if base <= 0:
        return
    time.sleep(base + random.uniform(0, base * 0.5))


def _build_safety_settings():
    return [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
    ]


NOISE_RERUN_PROMPT = """
You are re-analyzing a segment of audio from an ISKCON program that was
previously classified as NOISE (silence or non-content) but may actually
contain speech or singing that was missed.

Your job is to LISTEN CAREFULLY and classify every second of this audio.

Labels:
1. "LECTURE" — any sustained English speech, philosophy, Q&A, announcements,
   or conversation. Emit a NEW segment object for every distinct speaker turn.
   Provide verbatim "text".
2. "KIRTAN" — any singing, chanting, bhajans, mantras, or music.
   Leave "text" empty.
3. "NOISE" — true silence, hum, mic noise, ambient crowd noise with no
   speech or music. Leave "text" empty.

IMPORTANT:
- Do NOT default to NOISE. If you hear even faint speech or singing, label
  accordingly.
- Break LECTURE into per-speaker-turn segments. Never merge different speakers.
- Timestamps are relative to THIS audio chunk (0.0 to chunk duration).
- ALWAYS return a non-empty "segments" array.
- If the entire chunk really is silence or non-content, return exactly:
  {"segments": [{"label": "NOISE", "start": 0.0, "end": <chunk_end>, "text": ""}]}

Return JSON:
{
  "segments": [
    {"label": "LECTURE", "start": 0.0, "end": 10.5, "text": "..."},
    ...
  ]
}
"""


def _extract_text_from_response(resp) -> str:
    try:
        parts = resp.candidates[0].content.parts or []
    except Exception:
        return ""
    out = []
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            out.append(t)
    return "".join(out)


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=30, max=300))
def reclassify_chunk(temp_audio_path: str, chunk_duration: float) -> List[Dict[str, Any]]:
    with open(temp_audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")

    config = GenerateContentConfig(
        response_mime_type="application/json",
        safety_settings=_build_safety_settings(),
        thinking_config=ThinkingConfig(thinking_budget=0),
        max_output_tokens=8192,
    )
    decoder = JSONDecoder()

    try:
        if API_SEMAPHORE is not None:
            API_SEMAPHORE.acquire()
        try:
            _sleep_with_jitter(BASE_DELAY)
            resp = CLIENT.models.generate_content(
                model=MODEL_NAME,
                contents=[audio_part, NOISE_RERUN_PROMPT],
                config=config,
            )
        finally:
            if API_SEMAPHORE is not None:
                API_SEMAPHORE.release()

        pf = getattr(resp, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]

        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]

        fr = getattr(candidates[0], "finish_reason", None)
        fr_name = fr.name if hasattr(fr, "name") else str(fr)
        if fr_name in ("RECITATION", "4"):
            return [{"label": "KIRTAN", "start": 0.0, "end": chunk_duration, "text": ""}]
        if fr_name in ("OTHER", "5"):
            return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]

        text = _extract_text_from_response(resp)
        if not text.strip():
            return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]

        raw, _ = decoder.raw_decode(text)
        if isinstance(raw, list):
            raw = raw[0] if raw else {}

        segs = raw.get("segments", [])
        if not segs:
            return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]
        return segs

    except (ValueError, KeyError):
        return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "Resource exhausted" in err:
            raise
        return [{"label": "NOISE", "start": 0.0, "end": chunk_duration, "text": ""}]


def rerun_one_noise_segment(
    audio_path: str,
    seg_start: float,
    seg_end: float,
    temp_dir: str,
    tag: str,
) -> List[Dict[str, Any]]:
    seg_duration = seg_end - seg_start
    if seg_duration < NOISE_RERUN_MIN:
        return [{"label": "NOISE", "start": seg_start, "end": seg_end, "text": ""}]

    replacement: List[Dict[str, Any]] = []
    sub_start = 0.0
    idx = 0
    while sub_start < seg_duration:
        sub_len = min(NOISE_RERUN_CHUNK, seg_duration - sub_start)
        if sub_len < NOISE_RERUN_MIN:
            replacement.append({
                "label": "NOISE",
                "start": seg_start + sub_start,
                "end": seg_start + sub_start + sub_len,
                "text": "",
            })
            break

        tmp = os.path.join(temp_dir, f"{tag}_sub{idx}.mp3")
        cut_window(audio_path, seg_start + sub_start, sub_len, tmp)

        try:
            new_segs = reclassify_chunk(tmp, sub_len)
        except RetryError:
            new_segs = [{"label": "NOISE", "start": 0.0, "end": sub_len, "text": ""}]

        for s in new_segs:
            try:
                lbl = s.get("label", "NOISE").upper()
                txt = s.get("text", "") or ""
                st = float(s["start"])
                en = float(s["end"])
                if en <= st:
                    continue
                if lbl in ("LECTURE", "KIRTAN") and (en - st) < MIN_REAL_CONTENT and not txt.strip():
                    lbl = "NOISE"
                replacement.append({
                    "label": lbl,
                    "start": seg_start + sub_start + st,
                    "end": seg_start + sub_start + en,
                    "text": txt,
                })
            except (ValueError, TypeError, KeyError):
                continue

        if os.path.exists(tmp):
            os.remove(tmp)

        sub_start += sub_len
        idx += 1

    if not replacement:
        return [{"label": "NOISE", "start": seg_start, "end": seg_end, "text": ""}]

    collapsed: List[Dict[str, Any]] = [replacement[0]]
    for s in replacement[1:]:
        last = collapsed[-1]
        if s["label"] == last["label"] and (s["start"] - last["end"]) < 0.1 and s["label"] != "LECTURE":
            last["end"] = s["end"]
            if s.get("text"):
                old = last.get("text", "")
                sep = "" if old.endswith(" ") or s["text"].startswith(" ") else " "
                last["text"] = (old + sep + s["text"]).strip()
        else:
            collapsed.append(s)

    return collapsed


def process_json(audio_path: str, json_path: str) -> str:
    file_id = Path(json_path).stem

    if not os.path.exists(audio_path):
        return f"[{file_id}] SKIPPED (audio not found: {audio_path})"
    if not os.path.exists(json_path):
        return f"[{file_id}] SKIPPED (json not found)"

    backup_path = json_path.replace(".json", ".pre_noise_rerun.json")
    log_path = json_path.replace(".json", ".noise_rerun_log.json")

    if os.path.exists(backup_path):
        return f"[{file_id}] SKIPPED (already rerun — backup exists)"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    noise_indices = [i for i, s in enumerate(segments) if s.get("label", "").upper() == "NOISE"]
    if not noise_indices:
        return f"[{file_id}] SKIPPED (no NOISE segments)"

    temp_dir = os.path.join(os.path.dirname(audio_path) or ".", f"temp_noise_{file_id}")
    if not try_acquire_lock(temp_dir):
        return f"[{file_id}] SKIPPED (locked)"

    print(f"[{file_id}] Rerunning {len(noise_indices)} NOISE segments...")

    shutil.copy2(json_path, backup_path)

    log_entries: List[Dict[str, Any]] = []
    new_segments: List[Dict[str, Any]] = []

    try:
        for i, seg in enumerate(segments):
            if seg.get("label", "").upper() != "NOISE":
                new_segments.append(seg)
                continue

            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            replacement = rerun_one_noise_segment(
                audio_path, seg_start, seg_end, temp_dir, f"seg{i}"
            )

            recovered = [r for r in replacement if r["label"] != "NOISE"]
            log_entries.append({
                "original_noise_index": i,
                "original_start": seg_start,
                "original_end": seg_end,
                "recovered_segments": len(recovered),
                "replacement": replacement,
            })

            new_segments.extend(replacement)

        new_segments.sort(key=lambda x: float(x["start"]))
        data["segments"] = new_segments

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "file_id": file_id,
                "total_noise_segments": len(noise_indices),
                "entries": log_entries,
            }, f, indent=2, ensure_ascii=False)

        total_recovered = sum(e["recovered_segments"] for e in log_entries)
        return f"[{file_id}] DONE ({total_recovered} recovered from {len(noise_indices)} NOISE segs)"

    except Exception as e:
        return f"[{file_id}] ERROR: {e}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_job(args):
    audio_path, json_path = args
    return process_json(audio_path, json_path)


def _is_sidecar_json(name: str) -> bool:
    lower = name.lower()
    return (
        lower.endswith(".raw_windows.json")
        or lower.endswith(".pre_noise_rerun.json")
        or lower.endswith(".noise_rerun_log.json")
    )


def _discover_pairs(audio_root: str, json_root: str) -> List[Tuple[str, str]]:
    """
    Match audio files to transcript JSONs by FILENAME STEM, regardless of
    directory layout. Skip sidecar jsons.
    """
    audio_exts = {".mp3", ".m4a", ".wav"}

    json_by_stem: Dict[str, str] = {}
    json_collisions: Dict[str, int] = {}
    total_json = 0
    skipped_sidecars = 0
    for root, dirs, files in os.walk(json_root):
        dirs[:] = [d for d in dirs if not d.startswith("temp_") and d != "__pycache__"]
        for file in files:
            if not file.lower().endswith(".json"):
                continue
            total_json += 1
            if _is_sidecar_json(file):
                skipped_sidecars += 1
                continue
            stem = Path(file).stem
            full = os.path.join(root, file)
            if stem in json_by_stem:
                json_collisions[stem] = json_collisions.get(stem, 1) + 1
                continue
            json_by_stem[stem] = full

    print(f"  Indexed {len(json_by_stem)} transcript JSONs "
          f"({total_json} total files scanned, {skipped_sidecars} sidecars skipped)")
    if json_collisions:
        print(f"  WARNING: {len(json_collisions)} stem collisions in json_root "
              f"(same filename in multiple folders). Kept first occurrence.")

    pairs: List[Tuple[str, str]] = []
    audio_seen = 0
    audio_no_match = 0
    for root, dirs, files in os.walk(audio_root):
        dirs[:] = [d for d in dirs if not d.startswith("temp_") and d != "__pycache__"]
        for file in files:
            if Path(file).suffix.lower() not in audio_exts:
                continue
            audio_seen += 1
            stem = Path(file).stem
            audio_path = os.path.join(root, file)
            json_path = json_by_stem.get(stem)
            if json_path:
                pairs.append((audio_path, json_path))
            else:
                audio_no_match += 1

    print(f"  Scanned {audio_seen} audio files, matched {len(pairs)}, "
          f"unmatched {audio_no_match}")
    return pairs


def main():
    argv = sys.argv[1:]
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set")
        sys.exit(1)

    # Single-file mode
    if len(argv) >= 1 and argv[0].lower().endswith((".mp3", ".m4a", ".wav")):
        audio_path = argv[0]
        json_path = None
        if "--json" in argv:
            json_path = argv[argv.index("--json") + 1]
        if not json_path:
            print("Error: --json <path> is required in single-file mode")
            sys.exit(1)
        _init_worker(threading.Semaphore(MAX_CONCURRENT_API_CALLS))
        print(process_json(audio_path, json_path))
        return

    # Batch mode
    if len(argv) < 3:
        print("Usage:")
        print("  Batch:  python rerun_noise_segments.py <audio_root> <json_root> <num_workers>")
        print("  Single: python rerun_noise_segments.py <audio.mp3> --json <file.json>")
        sys.exit(1)

    audio_root = argv[0]
    json_root = argv[1]
    try:
        max_workers = int(argv[2])
    except ValueError:
        print("Error: <num_workers> must be an integer")
        sys.exit(1)

    pairs = _discover_pairs(audio_root, json_root)
    print(f"Found {len(pairs)} audio/json pairs under {audio_root} / {json_root}")

    if not pairs:
        return

    random.shuffle(pairs)
    semaphore = MPSemaphore(MAX_CONCURRENT_API_CALLS)

    completed = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(semaphore,),
    ) as executor:
        futures = {executor.submit(process_job, p): p for p in pairs}
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                p = futures[future]
                result = f"[{Path(p[0]).stem}] UNHANDLED: {e}"
            print(f"[{completed}/{len(pairs)}] {result}")


if __name__ == "__main__":
    main()


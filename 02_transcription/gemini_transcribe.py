"""
gemini_transcribe.py — Gemini 3 Flash Preview (AI Studio) audio segmenter.

Windows audio into 54s chunks and classifies each second as LECTURE / KIRTAN / NOISE,
preserving speaker-turn boundaries within LECTURE segments. Diarization (who is
speaking) is handled downstream by PyAnnote + WavLM + speaker_merge.py.

Batch mode (multi-file, process pool):
    python gemini_transcribe.py <input_audio_root> <output_json_root> <num_workers>

Single-file mode (legacy bash loop compatibility with intra-file window parallelism):
    python gemini_transcribe.py <audio.mp3> --output_dir <dir> [--threads N]

Env:
    GOOGLE_API_KEY    AI Studio API key
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
from typing import List, Dict, Any
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

WINDOW_SIZE = 54.0
MIN_NOISE_DURATION = 21.6
LOCK_STALE_SECONDS = 3600
MAX_CONCURRENT_API_CALLS = 4
BASE_DELAY = 0
WRITE_RAW_SIDECAR = True

API_SEMAPHORE = None
CLIENT: genai.Client = None


def _init_worker(semaphore):
    global API_SEMAPHORE, CLIENT
    API_SEMAPHORE = semaphore
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in the worker environment")
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
                    print(f"  -> Stale lock detected ({age:.0f}s old), reclaiming: {temp_dir}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    os.mkdir(temp_dir)
                    return True
            except OSError:
                pass
            return False
        raise


def get_audio_duration(path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd).decode().strip()
        return float(out)
    except Exception as e:
        print(f"Error getting duration for {path}: {e}")
        return 0.0


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


def _make_fallback(label: str = "NOISE") -> Dict[str, Any]:
    return {"segments": [{"label": label, "start": 0.0, "end": WINDOW_SIZE, "text": ""}]}


def _sleep_with_jitter(base: float):
    if base <= 0:
        return
    jitter = random.uniform(0, base * 0.5)
    total = base + jitter
    print(f"  -> Waiting {total:.1f}s before next API call...")
    time.sleep(total)


def _build_safety_settings():
    return [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
    ]


PROMPT = """
You are analyzing a 54-second audio chunk from an ISKCON program.

TASK: Segment the audio timeline by CONTENT TYPE and SPEAKER TURNS.

For every second of audio, classify it as one of the following:

1. "LECTURE": Sustained English speech, philosophy, or Q&A.
   **CRITICAL**: Every speaker change MUST start a new segment object, even if
   the gap between turns is under one second. Never concatenate text from two
   different speakers into one segment.

2. "KIRTAN": Singing, chanting, bhajans (instruments + vocals).
   (NOTE: Brief announcements or "Hari Bol" interjections during music are still KIRTAN).

3. "NOISE": Silence, hum, mic noise, crowd noise > 20 seconds.

Instructions:
- If "LECTURE": Provide verbatim "text". Emit a NEW segment object for every
  distinct speaker turn. Do NOT merge turns even when they are back-to-back.
- If "KIRTAN" or "NOISE": Leave "text" empty.
- Return start/end times relative to this chunk (0.0 to 54.0).
- If the ENTIRE chunk is KIRTAN, return exactly one segment:
  {"label": "KIRTAN", "start": 0.0, "end": 54.0, "text": ""}
- If the ENTIRE chunk is silence/noise, return exactly one segment:
  {"label": "NOISE",  "start": 0.0, "end": 54.0, "text": ""}
- ALWAYS return a non-empty "segments" array.

Return JSON:
{
  "segments": [
     {"label": "LECTURE", "start": 0.0, "end": 10.5, "text": "Question: Why is..."},
     {"label": "LECTURE", "start": 10.5, "end": 45.0, "text": "Answer: Because Krsna says..."},
     {"label": "KIRTAN", "start": 45.0, "end": 54.0, "text": ""}
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


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=30, max=300),
)
def segment_mixed_content(temp_audio_path: str) -> Dict[str, Any]:
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
                contents=[audio_part, PROMPT],
                config=config,
            )
        finally:
            if API_SEMAPHORE is not None:
                API_SEMAPHORE.release()

        pf = getattr(resp, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            print(f"  -> BLOCKED by safety filter: {pf.block_reason}")
            return _make_fallback("KIRTAN")

        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            fr = getattr(candidates[0], "finish_reason", None)
            fr_name = fr.name if hasattr(fr, "name") else str(fr)

            if fr_name in ("RECITATION", "4"):
                print("  -> RECITATION block — marking as KIRTAN")
                return _make_fallback("KIRTAN")

            if fr_name in ("OTHER", "5"):
                print("  -> finish_reason OTHER — marking as NOISE")
                return _make_fallback("NOISE")

            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                print(f"  -> Empty parts (finish_reason: {fr_name}) — marking as NOISE")
                return _make_fallback("NOISE")
        else:
            print("  -> No candidates returned — marking as NOISE")
            return _make_fallback("NOISE")

        text = _extract_text_from_response(resp)
        if not text.strip():
            print("  -> Empty response text — marking as NOISE")
            return _make_fallback("NOISE")

        raw, _ = decoder.raw_decode(text)
        if isinstance(raw, list):
            raw = raw[0] if raw else {}

    except (ValueError, KeyError) as e:
        print(f"  -> JSON parse error: {e}")
        raw = {}
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "Resource exhausted" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            print("  -> Rate limited (429), will retry with backoff...")
            raise
        print(f"  -> Gemini error: {e}")
        raise

    return {"segments": raw.get("segments", [])}


def stitch_segments(all_windows: List[Dict]) -> List[Dict]:
    """
    Stitch per-window segments into absolute-time segments.

    Merge policy:
      - KIRTAN: merge adjacent KIRTAN with gap < 2.0s (healing cuts in music).
      - LECTURE: merge ONLY across window boundaries with gap < 2.0s. Never
        merge intra-window LECTURE segments — those are speaker turns.
      - NOISE < MIN_NOISE_DURATION is promoted to the previous label.
    """
    final_segments: List[Dict] = []
    all_windows.sort(key=lambda x: x["window_start"])

    for window in all_windows:
        win_start = window["window_start"]

        if final_segments:
            last_seg_end = final_segments[-1]["end"]
            gap = win_start - last_seg_end
            if gap > 1.0:
                print(f"  -> Detected GAP of {gap:.1f}s at {last_seg_end:.1f}s. Auto-filling...")
                final_segments.append({
                    "label": final_segments[-1]["label"],
                    "start": last_seg_end,
                    "end": win_start,
                    "text": "[MISSING AUDIO - GAP FILLED]",
                })

        for seg in window["segments"]:
            try:
                label = seg.get("label", "NOISE").upper()
                text = seg.get("text", "") or ""
                start = float(seg["start"])
                end = float(seg["end"])

                if start >= end:
                    continue

                duration = end - start

                if label == "NOISE" and duration < MIN_NOISE_DURATION:
                    label = final_segments[-1]["label"] if final_segments else "LECTURE"

                abs_start = win_start + start
                abs_end = win_start + end

                if final_segments:
                    last_seg = final_segments[-1]
                    seg_gap = abs_start - last_seg["end"]

                    should_merge = False
                    if label == last_seg["label"]:
                        if label == "KIRTAN":
                            should_merge = seg_gap < 2.0
                        elif label == "LECTURE":
                            mod = abs_start % WINDOW_SIZE
                            at_boundary = mod < 2.0 or mod > (WINDOW_SIZE - 2.0)
                            if at_boundary and seg_gap < 2.0:
                                should_merge = True

                    if should_merge:
                        final_segments[-1]["end"] = abs_end
                        if text:
                            old = final_segments[-1].get("text", "")
                            sep = "" if old.endswith(" ") or text.startswith(" ") else " "
                            final_segments[-1]["text"] = old + sep + text
                    else:
                        final_segments.append({
                            "label": label, "start": abs_start,
                            "end": abs_end, "text": text,
                        })
                else:
                    final_segments.append({
                        "label": label, "start": abs_start,
                        "end": abs_end, "text": text,
                    })

            except (ValueError, TypeError, KeyError):
                # Skip a single malformed segment (missing/non-numeric
                # start/end) instead of failing the whole file. A lone bad
                # segment from Gemini was previously surfacing as the
                # "CRITICAL ERROR: 'end'" that discarded an entire transcript.
                continue

    return final_segments


def _process_window(args):
    audio_path, start, current_len, temp_dir, window_index, file_id = args
    temp_file = os.path.join(temp_dir, f"chunk_{window_index}.mp3")
    cut_window(audio_path, start, current_len, temp_file)

    try:
        result = segment_mixed_content(temp_file)
    except RetryError as e:
        err = e.last_attempt.exception()
        print(f"[{file_id}] Window {window_index} FAILED after retries: {err}")
        result = _make_fallback("NOISE")

    if not result.get("segments"):
        print(f"  -> 0 segments for window {window_index}, defaulting to NOISE")
        result = _make_fallback("NOISE")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return {"window_start": start, "segments": result.get("segments", [])}


def process_file_mirror(args):
    audio_path, relative_path, audio_root, output_root = args
    file_id = Path(audio_path).stem

    output_dir = os.path.join(output_root, os.path.dirname(relative_path))
    out_json_path = os.path.join(output_dir, f"{file_id}.json")
    raw_sidecar_path = os.path.join(output_dir, f"{file_id}.raw_windows.json")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(out_json_path):
        return f"[{file_id}] SKIPPED (Exists)"

    input_dir = os.path.dirname(audio_path)
    temp_dir = os.path.join(input_dir, f"temp_{file_id}")

    if not try_acquire_lock(temp_dir):
        return f"[{file_id}] SKIPPED (Another process is working)"

    duration = get_audio_duration(audio_path)
    if duration <= 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return f"[{file_id}] FAILED (Duration 0)"

    print(f"[{file_id}] STARTING ({duration:.1f}s)...")

    MIN_WINDOW = 1.0
    window_plans = []
    start = 0.0
    window_index = 0
    while start < duration:
        current_len = min(WINDOW_SIZE, duration - start)
        if current_len < MIN_WINDOW:
            break
        window_plans.append((audio_path, start, current_len, temp_dir, window_index, file_id))
        start += current_len
        window_index += 1

    try:
        raw_windows: List[Dict] = []
        for plan in window_plans:
            raw_windows.append(_process_window(plan))

        if WRITE_RAW_SIDECAR:
            with open(raw_sidecar_path, "w", encoding="utf-8") as f:
                json.dump(raw_windows, f, indent=2, ensure_ascii=False)

        stitched_output = stitch_segments(raw_windows)

        final_data = {
            "file_id": file_id,
            "total_duration": duration,
            "segments": stitched_output,
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        return f"[{file_id}] DONE -> {out_json_path}"

    except Exception as e:
        return f"[{file_id}] CRITICAL ERROR: {e}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_single_file(audio_path: str, output_dir: str, threads: int):
    os.makedirs(output_dir, exist_ok=True)
    file_id = Path(audio_path).stem
    out_json_path = os.path.join(output_dir, f"{file_id}.json")
    raw_sidecar_path = os.path.join(output_dir, f"{file_id}.raw_windows.json")

    if os.path.exists(out_json_path):
        print(f"[{file_id}] SKIPPED (Exists)")
        return

    input_dir = os.path.dirname(audio_path) or "."
    temp_dir = os.path.join(input_dir, f"temp_{file_id}")
    if not try_acquire_lock(temp_dir):
        print(f"[{file_id}] SKIPPED (Another process is working)")
        return

    _init_worker(threading.Semaphore(MAX_CONCURRENT_API_CALLS))

    duration = get_audio_duration(audio_path)
    if duration <= 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[{file_id}] FAILED (Duration 0)")
        return

    print(f"[{file_id}] STARTING ({duration:.1f}s) with {threads} threads...")

    MIN_WINDOW = 1.0
    window_plans = []
    start = 0.0
    window_index = 0
    while start < duration:
        current_len = min(WINDOW_SIZE, duration - start)
        if current_len < MIN_WINDOW:
            break
        window_plans.append((audio_path, start, current_len, temp_dir, window_index, file_id))
        start += current_len
        window_index += 1

    try:
        raw_windows: List[Dict] = [None] * len(window_plans)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {ex.submit(_process_window, p): i for i, p in enumerate(window_plans)}
            for fut in as_completed(futs):
                i = futs[fut]
                raw_windows[i] = fut.result()

        if WRITE_RAW_SIDECAR:
            with open(raw_sidecar_path, "w", encoding="utf-8") as f:
                json.dump(raw_windows, f, indent=2, ensure_ascii=False)

        stitched_output = stitch_segments(raw_windows)

        final_data = {
            "file_id": file_id,
            "total_duration": duration,
            "segments": stitched_output,
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"[{file_id}] DONE -> {out_json_path}")

    except Exception as e:
        print(f"[{file_id}] CRITICAL ERROR: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    argv = sys.argv[1:]

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set")
        sys.exit(1)

    # Single-file mode
    if len(argv) >= 1 and argv[0].lower().endswith((".mp3", ".m4a", ".wav")):
        audio_path = argv[0]
        output_dir = None
        threads = MAX_CONCURRENT_API_CALLS
        if "--output_dir" in argv:
            i = argv.index("--output_dir")
            output_dir = argv[i + 1]
        if "--threads" in argv:
            i = argv.index("--threads")
            try:
                threads = int(argv[i + 1])
            except ValueError:
                print("Error: --threads must be an integer")
                sys.exit(1)
        if not output_dir:
            stem = Path(audio_path).stem
            output_dir = f"FTS_MP3_gemini/{stem}"
        _process_single_file(audio_path, output_dir, threads)
        return

    # Batch mode
    if len(argv) < 3:
        print("Usage:")
        print("  Batch:  python gemini_transcribe.py <input_audio_root> <output_json_root> <num_workers>")
        print("  Single: python gemini_transcribe.py <audio.mp3> --output_dir <dir> [--threads N]")
        sys.exit(1)

    input_root = argv[0]
    output_root = argv[1]
    try:
        max_workers = int(argv[2])
    except ValueError:
        print("Error: <num_workers> must be an integer")
        sys.exit(1)

    semaphore = MPSemaphore(MAX_CONCURRENT_API_CALLS)

    audio_extensions = {".mp3", ".m4a"}
    tasks = []

    print(f"Scanning {input_root}...")
    print(f"Model: {MODEL_NAME} | SDK: google.genai (AI Studio)")
    print(f"Workers: {max_workers} | API concurrency cap: {MAX_CONCURRENT_API_CALLS}")
    print("Retries: up to 10 attempts with 30s-300s exponential backoff")

    for root, dirs, files in os.walk(input_root):
        dirs[:] = [
            d for d in dirs
            if not d.startswith("temp_")
            and not d.endswith("_temp")
            and d != "__pycache__"
        ]
        for file in files:
            if Path(file).suffix.lower() not in audio_extensions:
                continue
            if file.startswith("chunk_"):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, input_root)
            tasks.append((full_path, rel_path, input_root, output_root))

    print(f"Found {len(tasks)} files. Processing with {max_workers} workers...")
    random.shuffle(tasks)

    completed = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(semaphore,),
    ) as executor:
        futures = {executor.submit(process_file_mirror, task): task for task in tasks}
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                task = futures[future]
                result = f"[{Path(task[0]).stem}] UNHANDLED EXCEPTION: {e}"
                failed += 1
            if "CRITICAL ERROR" in result or "FAILED" in result:
                failed += 1
            print(f"[{completed}/{len(tasks)}] {result}")

    print("\n" + "=" * 60)
    print(f"FINISHED: {completed} processed, {failed} failed, {len(tasks) - completed} remaining")
    print("=" * 60)


if __name__ == "__main__":
    main()

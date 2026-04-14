"""
gemini_transcribe.py - Gemini Flash Audio Transcription (54s Chunks)

Splits lecture audio into 54-second chunks grouped by kirtan boundaries,
sends each to Gemini Flash for high-quality transcription with speaker labels.

Usage:
    python gemini_transcribe.py <audio.mp3> [--output_dir <dir>]

Environment:
    GOOGLE_API_KEY      API key for Google Generative AI
    KIRTANTIMES_DIR     Kirtan segment JSONs (to skip non-lecture audio)
"""
import os, sys, json, subprocess, tempfile, re, time
from pathlib import Path
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME = "gemini-3-flash"
MAX_CHUNK_SEC = 54.0
GAP_THRESHOLD = 5.4

genai.configure(api_key=API_KEY)

def get_duration(path):
    r = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=noprint_wrappers=1:nokey=1", path])
    return float(r.decode().strip())

def extract_chunk(audio_path, start, duration, tmp_dir):
    out = os.path.join(tmp_dir, f"chunk_{int(start)}.mp3")
    subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ss", str(start), "-t", str(duration),
                     "-acodec", "libmp3lame", "-q:a", "4", out], capture_output=True, check=True)
    return out

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5))
def transcribe_chunk(audio_bytes):
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([
        {"mime_type": "audio/mpeg", "data": audio_bytes},
        "Transcribe this audio. Include speaker labels if multiple speakers. "
        "Return JSON array: [{\"speaker\": \"name\", \"text\": \"...\", \"start\": seconds, \"end\": seconds}]"
    ])
    text = response.text.strip()
    if text.startswith("```"): text = re.sub(r"^```\w*\n?", "", text).rstrip("`").strip()
    return json.loads(text)

def main():
    audio_path = sys.argv[1]
    stem = Path(audio_path).stem
    output_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--output_dir" else f"FTS_MP3_gemini/{stem}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "transcript.json")
    if os.path.exists(output_path): print(f"[SKIP] {output_path} exists"); return

    duration = get_duration(audio_path)
    all_segments = []
    with tempfile.TemporaryDirectory() as tmp:
        t = 0.0
        while t < duration:
            chunk_dur = min(MAX_CHUNK_SEC, duration - t)
            if chunk_dur < 1.0: break
            chunk_path = extract_chunk(audio_path, t, chunk_dur, tmp)
            with open(chunk_path, "rb") as f: audio_bytes = f.read()
            try:
                segs = transcribe_chunk(audio_bytes)
                for s in segs:
                    s["start"] = s.get("start", 0) + t
                    s["end"] = s.get("end", 0) + t
                all_segments.extend(segs)
                print(f"  {t:.0f}-{t+chunk_dur:.0f}s: {len(segs)} segments")
            except Exception as e:
                print(f"  {t:.0f}s: ERROR {e}")
            t += chunk_dur

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {stem}: {len(all_segments)} segments -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python gemini_transcribe.py <audio.mp3>"); sys.exit(1)
    main()

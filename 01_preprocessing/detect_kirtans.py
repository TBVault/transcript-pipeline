"""
detect_kirtans.py - Classify Audio Segments as LECTURE vs KIRTAN

Uses spectral centroid, RMS energy, and zero-crossing rate to detect
kirtan/bhajan musical interludes. Outputs a JSON of labeled time windows.

Usage:
    python detect_kirtans.py <lecture_id>

Environment:
    AUDIO_ROOT          Path to MP3 files
    KIRTANTIMES_DIR     Output directory for kirtan segment JSONs
"""

import os, sys, json
import numpy as np
import librosa

AUDIO_ROOT = os.getenv("AUDIO_ROOT", ".")
KIRTANTIMES_DIR = os.getenv("KIRTANTIMES_DIR", "FTS_MP3_kirtantimes")
WINDOW_SEC = 30
HOP_SEC = 15
ENERGY_THRESHOLD = 0.02
CENTROID_HIGH = 3000

def analyze_window(y, sr):
    rms = np.sqrt(np.mean(y**2))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return {"rms": float(rms), "centroid": float(centroid), "zcr": float(zcr)}

def classify_segment(features):
    if features["rms"] < ENERGY_THRESHOLD:
        return "SILENCE"
    if features["centroid"] > CENTROID_HIGH and features["zcr"] > 0.1:
        return "KIRTAN"
    return "LECTURE"

def main():
    lecture_id = sys.argv[1]
    audio_path = os.path.join(AUDIO_ROOT, f"{lecture_id}.mp3")
    output_path = os.path.join(KIRTANTIMES_DIR, f"{lecture_id}.json")
    os.makedirs(KIRTANTIMES_DIR, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    segments = []
    t = 0.0
    while t < duration:
        end = min(t + WINDOW_SEC, duration)
        start_sample, end_sample = int(t * sr), int(end * sr)
        window = y[start_sample:end_sample]
        if len(window) > sr * 0.5:
            features = analyze_window(window, sr)
            label = classify_segment(features)
            segments.append({"start": round(t, 1), "end": round(end, 1), "label": label, **features})
        t += HOP_SEC

    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)
    lec = sum(1 for s in segments if s["label"] == "LECTURE")
    kir = sum(1 for s in segments if s["label"] == "KIRTAN")
    print(f"[DONE] {lecture_id}: {lec} lecture / {kir} kirtan windows -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python detect_kirtans.py <lecture_id>"); sys.exit(1)
    main()

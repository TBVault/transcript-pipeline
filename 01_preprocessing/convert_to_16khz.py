"""
convert_to_16khz.py - Batch Resample to 16kHz Mono WAV

Usage: python convert_to_16khz.py <input_dir/> <output_dir/>
"""
import os, sys, torchaudio
from pathlib import Path

TARGET_SR = 16000

def main():
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(list(in_dir.glob("*.mp3")) + list(in_dir.glob("*.wav"))):
        out = out_dir / f"{f.stem}.wav"
        if out.exists(): continue
        try:
            wav, sr = torchaudio.load(str(f))
            if sr != TARGET_SR: wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            torchaudio.save(str(out), wav, TARGET_SR)
            print(f"OK: {f.name} -> {wav.shape[1]/TARGET_SR:.1f}s")
        except Exception as e: print(f"ERR: {f.name} -> {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3: print("Usage: python convert_to_16khz.py <in> <out>"); sys.exit(1)
    main()

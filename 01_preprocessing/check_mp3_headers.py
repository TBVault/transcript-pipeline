"""
check_mp3_headers.py - Fix Corrupted MP3 Headers via ffmpeg

Usage: python check_mp3_headers.py <mp3_folder/> [--fix]
"""
import os, sys, subprocess
from pathlib import Path
import torchaudio

def check_header(path):
    try: torchaudio.info(str(path)); return True
    except: return False

def fix_mp3(path):
    tmp = str(path) + ".tmp.mp3"
    r = subprocess.run(["ffmpeg", "-y", "-i", str(path), "-acodec", "libmp3lame", "-q:a", "2", tmp], capture_output=True)
    if r.returncode == 0: os.replace(tmp, str(path)); return True
    if os.path.exists(tmp): os.remove(tmp)
    return False

def main():
    folder = Path(sys.argv[1])
    do_fix = "--fix" in sys.argv
    mp3s = sorted(folder.glob("*.mp3"))
    broken = [p for p in mp3s if not check_header(p)]
    print(f"{len(mp3s)-len(broken)}/{len(mp3s)} OK, {len(broken)} broken")
    if do_fix:
        for p in broken:
            print(f"{'FIXED' if fix_mp3(p) else 'FAILED'}: {p.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python check_mp3_headers.py <folder/> [--fix]"); sys.exit(1)
    main()

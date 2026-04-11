"""
batch_parallel.py - Parallel Speaker Merge

Finds IDs with both Whisper + PyAnnote outputs, runs in parallel.

Usage: python batch_parallel.py --output_dir v2/ [--workers N]
"""
import os, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from speaker_merge import process_file

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--whisper_dir", default=os.getenv("WHISPER_OUTPUT_DIR", "FTS_MP3_whisper"))
    p.add_argument("--pyannote_dir", default=os.getenv("PYANNOTE_OUTPUT_DIR", "FTS_MP3_pyannote"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    wd, pd_, od = Path(args.whisper_dir), Path(args.pyannote_dir), Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    ids = {s.name for s in wd.iterdir() if s.is_dir() and (s/"transcript.json").exists()}
    ids &= {t.stem for t in pd_.glob("*.txt")}
    remaining = [i for i in sorted(ids) if not (od/f"{i}.json").exists()]
    print(f"Processing {len(remaining)}/{len(ids)}...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_file, fid, str(od)): fid for fid in remaining}
        for fut in as_completed(futs):
            try: fut.result()
            except Exception as e: print(f"{futs[fut]}: {e}")

if __name__ == "__main__": main()

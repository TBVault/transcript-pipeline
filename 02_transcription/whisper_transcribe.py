"""
whisper_transcribe.py - WhisperX Word-Level Transcription

Runs WhisperX with forced alignment for precise word-level timestamps.
These timestamps are later fused with Gemini text via NW alignment.

Usage:
    python whisper_transcribe.py <audio.mp3> [--output_dir <dir>]

Environment:
    HF_TOKEN                For diarization model access (if --diarize)
    CUDA_VISIBLE_DEVICES    GPU selection
"""
import os, sys, json
from pathlib import Path
import whisperx, torch
import omegaconf
import omegaconf.base, omegaconf.nodes, omegaconf.basecontainer
# PyTorch 2.6 fix: allow omegaconf globals when loading pyannote VAD model
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.nodes.ValueNode,
    omegaconf.basecontainer.BaseContainer,
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
LANGUAGE = "en"
HF_TOKEN = os.getenv("HF_TOKEN", "")

def main():
    audio_path = sys.argv[1]
    stem = Path(audio_path).stem
    output_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--output_dir" else f"FTS_MP3_whisper/{stem}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "transcript.json")
    if os.path.exists(output_path): print(f"[SKIP] {output_path} exists"); return

    print(f"Loading WhisperX on {DEVICE}...")
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)

    print(f"Transcribing {audio_path}...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)

    print("Aligning...")
    align_model, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
    result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE)

    segments = []
    for seg in result["segments"]:
        entry = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        if "words" in seg:
            entry["words"] = [{"word": w["word"], "start": w.get("start", seg["start"]),
                               "end": w.get("end", seg["end"])} for w in seg["words"]]
        segments.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {stem}: {len(segments)} segments -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python whisper_transcribe.py <audio.mp3>"); sys.exit(1)
    main()

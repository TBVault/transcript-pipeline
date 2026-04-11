"""
diar.py - Minimal Single-File PyAnnote Diarization

Usage: python diar.py [audio_file]
"""
import sys, os, torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

HF_TOKEN = os.getenv("HF_TOKEN", "")
audio = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
with ProgressHook() as hook:
    output = pipe(audio, hook=hook)
for turn, _, spk in output.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{spk}")

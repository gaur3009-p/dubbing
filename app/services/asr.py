from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(
    "small",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)

def transcribe(audio_path):
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False
    )
    text = ""
    for seg in segments:
        text += seg.text
    return text.strip(), info.language

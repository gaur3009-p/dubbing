from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(
    "medium",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)

def transcribe_travel(audio_path):
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="This is a conversation."
    )
    text = ""
    for seg in segments:
        text += seg.text
    return text.strip(), info.language

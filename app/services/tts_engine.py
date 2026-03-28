import torch
import uuid
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

def speak(text):
    description = "A clear conversational voice."
    desc = desc_tokenizer(description, return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)
    audio = model.generate(
        input_ids=desc.input_ids,
        attention_mask=desc.attention_mask,
        prompt_input_ids=prompt.input_ids,
        prompt_attention_mask=prompt.attention_mask,
        do_sample=True
    )
    audio = audio.cpu().numpy().squeeze()
    file = f"speech_{uuid.uuid4()}.wav"
    sf.write(file, audio, model.config.sampling_rate)
    return file

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def translate(text, src, tgt):
    tokenizer.src_lang = src
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt)
    tokens = model.generate(
        **inputs,
        forced_bos_token_id=tgt_id,
        max_length=200
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

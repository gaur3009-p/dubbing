"""
Gemini 1.5 Flash streaming translation.
Replaces NLLB for lower TTFT (~80ms) while preserving conversational tone.

Set GEMINI_API_KEY env var. Falls back to NLLB (translator.py) automatically
for Indic scripts where Gemini may underperform (Sindhi, Sanskrit, Assamese).

Usage:
    translated = await translate_gemini(text, src_lang, "hin_Deva")
"""
import os
import asyncio
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Indic languages Gemini handles well
GEMINI_SUPPORTED = {
    "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "kan_Knda",
    "mal_Mlym", "mar_Deva", "guj_Gujr", "pan_Guru", "eng_Latn",
}

# Human-readable names for the prompt
LANG_NAMES = {
    "eng_Latn": "English",   "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",   "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam", "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati",  "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu",      "npi_Deva": "Nepali",
    "ory_Orya": "Odia",      "asm_Beng": "Assamese",
    "snd_Arab": "Sindhi",    "san_Deva": "Sanskrit",
}

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _model = genai.GenerativeModel("gemini-1.5-flash")
else:
    _model = None


async def translate_gemini(text: str, src_lang: str, tgt_code: str) -> str:
    """
    Async streaming translation via Gemini 1.5 Flash.
    Returns the full translated string (collects stream internally).
    Falls back to NLLB for unsupported language pairs.
    """
    if _model is None or tgt_code not in GEMINI_SUPPORTED:
        # Fallback to local NLLB
        from services.translator import translate as translate_nllb
        return translate_nllb(text, src_lang, tgt_code)

    tgt_name = LANG_NAMES.get(tgt_code, tgt_code)
    prompt = (
        f"Translate the following spoken utterance to {tgt_name}. "
        f"Preserve the conversational tone, pacing, and register. "
        f"Return ONLY the translation — no explanation, no quotes.\n\n{text}"
    )

    result = ""
    response = await _model.generate_content_async(prompt, stream=True)
    async for chunk in response:
        result += chunk.text

    return result.strip()


def translate_gemini_sync(text: str, src_lang: str, tgt_code: str) -> str:
    """Synchronous wrapper for use in thread-pool workers."""
    return asyncio.get_event_loop().run_until_complete(
        translate_gemini(text, src_lang, tgt_code)
    )

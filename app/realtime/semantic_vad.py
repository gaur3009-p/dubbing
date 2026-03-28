"""
Semantic VAD — predicts utterance completeness without relying solely on silence.
Drop-in upgrade for the silence-trigger logic in RollingBuffer.

Phase 1: simple heuristic (punctuation + length).
Phase 2: swap _is_complete_model() for a local BERT classifier or Deepgram Flux.
"""
import re

_TERMINAL = re.compile(r'[.!?]["\'»]?\s*$')
_QUESTION = re.compile(r'\?["\'»]?\s*$')


def is_utterance_complete(transcript: str, min_words: int = 4) -> bool:
    """
    Returns True when the transcript reads as a semantically complete thought.

    Args:
        transcript: Interim transcript from STT (may lack punctuation).
        min_words:  Minimum word count before triggering completion.
    """
    words = transcript.strip().split()
    if len(words) < min_words:
        return False

    # Heuristic 1: explicit terminal punctuation
    if _TERMINAL.search(transcript):
        return True

    # Heuristic 2: question pattern detected
    if _QUESTION.search(transcript):
        return True

    # Heuristic 3: long utterance with no continuation cue
    if len(words) >= 20:
        last = words[-1].lower().rstrip('.,;:')
        continuation_cues = {'and', 'but', 'or', 'so', 'because', 'although',
                              'however', 'also', 'then', 'when', 'if', 'that'}
        if last not in continuation_cues:
            return True

    return False


async def is_utterance_complete_api(transcript: str, api_key: str) -> bool:
    """
    Phase 2: call Deepgram Flux or a remote classifier for semantic endpointing.
    Replace the heuristic above with this once you have API access.
    """
    # Placeholder — integrate Deepgram Flux endpoint here
    return is_utterance_complete(transcript)

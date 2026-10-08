"""First-pass ASR with word timestamps; these times anchor the reference transcript.

Uses faster-whisper (https://github.com/SYSTRAN/faster-whisper) with its built-in Silero VAD filter,
which skips non-speech and reduces hallucinated text in long recordings. Long hearings are transcribed
in windows that are cut at silences, so memory use does not grow with hearing length and no word is split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

WINDOW_SEARCH_SECONDS = 60.0  # look this far either side of each target cut for a silence


def window_cuts(gaps: list[tuple[float, float]], duration: float, window_seconds: float) -> list[float]:
    """Cut times roughly every window_seconds, each moved to the longest nearby silence."""
    cuts = [0.0]
    target = window_seconds
    while target < duration - window_seconds / 4:
        nearby = [g for g in gaps if abs((g[0] + g[1]) / 2 - target) <= WINDOW_SEARCH_SECONDS
                  and (g[0] + g[1]) / 2 > cuts[-1]]
        if nearby:
            longest = max(nearby, key=lambda g: g[1] - g[0])
            cut = (longest[0] + longest[1]) / 2
        else:
            cut = target
        cuts.append(round(cut, 3))
        target = cut + window_seconds
    cuts.append(duration)
    return cuts


class WordTimestamper:
    def __init__(self, model: str, device: str, compute_type: str | None = None, beam_size: int = 5,
                 cpu_threads: int = 4, window_minutes: float = 20.0):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, cpu_threads=cpu_threads,
                                  compute_type=compute_type or ("float16" if device == "cuda" else "int8"))
        self.beam_size = beam_size
        self.window_seconds = window_minutes * 60

    def __call__(self, wav: Path, gaps: list[tuple[float, float]], duration: float) -> list[dict[str, Any]]:
        cuts = window_cuts(gaps, duration, self.window_seconds)
        words: list[dict[str, Any]] = []
        with sf.SoundFile(wav) as source:
            sr = source.samplerate
            for start, end in zip(cuts, cuts[1:]):
                source.seek(int(start * sr))
                audio = source.read(int(end * sr) - int(start * sr), dtype="float32")
                segments, _ = self.model.transcribe(
                    audio, language="en", beam_size=self.beam_size, word_timestamps=True,
                    vad_filter=True, condition_on_previous_text=False,
                )
                words.extend({"word": w.word.strip(), "start": round(start + w.start, 3),
                              "end": round(start + w.end, 3), "probability": round(w.probability, 3)}
                             for segment in segments for w in (segment.words or []) if w.word.strip())
        return words

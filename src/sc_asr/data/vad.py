"""Hearing-level speech/silence map from Silero VAD (https://github.com/snakers4/silero-vad).

Uses the ONNX copy of Silero bundled with faster-whisper (no torch). Audio is read in fixed blocks so
memory use does not grow with hearing length. The map is cached as JSON next to the other outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf

BLOCK_SECONDS = 600


def speech_segments(wav: Path) -> dict[str, Any]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=0)
    segments: list[list[float]] = []
    with sf.SoundFile(wav) as source:
        sr = source.samplerate
        offset = 0
        while True:
            block = source.read(BLOCK_SECONDS * sr, dtype="float32")
            if not len(block):
                break
            for stamp in get_speech_timestamps(block, options, sampling_rate=sr):
                start, end = (offset + stamp["start"]) / sr, (offset + stamp["end"]) / sr
                if segments and start - segments[-1][1] < 0.01:  # speech running across a block border
                    segments[-1][1] = round(end, 3)
                else:
                    segments.append([round(start, 3), round(end, 3)])
            offset += len(block)
        duration = offset / sr
    return {"duration": round(duration, 3), "speech": segments}


def load_speech_map(wav: Path, cache: Path) -> dict[str, Any]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    result = speech_segments(wav)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    return result


def silences(speech_map: dict[str, Any], min_length: float = 0.1) -> list[tuple[float, float]]:
    """Gaps between speech segments (including before the first and after the last), sorted by time."""
    gaps = []
    previous_end = 0.0
    for start, end in speech_map["speech"]:
        if start - previous_end >= min_length:
            gaps.append((previous_end, start))
        previous_end = end
    if speech_map["duration"] - previous_end >= min_length:
        gaps.append((previous_end, speech_map["duration"]))
    return gaps

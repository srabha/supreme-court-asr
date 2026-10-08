"""Align a hearing transcript to ASR word timestamps and cut it into verified chunks.

Cuts are only made inside runs of exactly matching words, at a pause, and are moved into the
nearest VAD silence (Whisper word end times tend to be early). Chunks are dropped when one side
has words the other lacks, which usually means the transcript skipped or added text. Labels
always come from the transcript; the ASR text is only used for timing and checks.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import DataPrepConfig
from .textnorm import normalize_word


@dataclass(frozen=True)
class Boundary:
    ref: int            # first reference token to the right of the cut
    hyp: int            # first ASR token to the right of the cut
    left_end: float     # end time of the chunk on the left
    right_start: float  # start time of the chunk on the right


def hyp_tokens(words: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    return [(token, w["start"], w["end"]) for w in words for token in normalize_word(w["word"])]


def find_boundaries(ref_origin: list[int], hyp: list[tuple[str, float, float]], opcodes,
                    gaps: list[tuple[float, float]], config: DataPrepConfig) -> list[Boundary]:
    """Cut points inside exact-match runs, each moved into the longest VAD silence near the ASR pause."""
    midpoints = [(a + b) / 2 for a, b in gaps]
    boundaries = []
    for tag, r0, r1, h0, h1 in opcodes:
        if tag != "equal" or r1 - r0 < config.min_anchor_words:
            continue
        for r, h in zip(range(r0 + 1, r1), range(h0 + 1, h1)):
            if ref_origin[r] == ref_origin[r - 1]:  # never split one written word ("1,000" -> one thousand)
                continue
            word_end, next_start = hyp[h - 1][2], hyp[h][1]
            if next_start - word_end < config.min_pause:
                continue
            lo = bisect_left(midpoints, word_end - config.snap_slack)
            candidates = []
            for k in range(lo, len(gaps)):
                if midpoints[k] > next_start + config.snap_slack:
                    break
                candidates.append(gaps[k])
            if not candidates:  # the ASR pause is not a real silence
                continue
            silence_start, silence_end = max(candidates, key=lambda g: g[1] - g[0])
            pad = min(config.edge_pad, (silence_end - silence_start) / 2)
            boundaries.append(Boundary(r, h, silence_start + pad, silence_end - pad))
    return boundaries


def longest_mismatch(ref: list[str], hyp: list[str]) -> int:
    """Largest word-count imbalance in any stretch between two matches (skipped/added text)."""
    from rapidfuzz.distance import Levenshtein

    worst = ref_run = hyp_run = 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag == "equal":
            worst = max(worst, abs(ref_run - hyp_run))
            ref_run = hyp_run = 0
        else:
            ref_run += i2 - i1
            hyp_run += j2 - j1
    return max(worst, abs(ref_run - hyp_run))


class SpeechChecker:
    """Silero VAD (https://github.com/snakers4/silero-vad) check on a single chunk.

    Uses the ONNX copy of Silero bundled with faster-whisper, so no torch is needed.
    """

    def __init__(self, sample_rate: int, edge_window: float):
        from faster_whisper.vad import VadOptions

        # Silero's own defaults for short audio, but without padding so edges are measured exactly.
        self.options = VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=0)
        self.sample_rate = sample_rate
        self.edge = int(edge_window * sample_rate)

    def __call__(self, audio: np.ndarray) -> dict[str, Any]:
        from faster_whisper.vad import get_speech_timestamps

        stamps = get_speech_timestamps(audio, self.options, sampling_rate=self.sample_rate)
        speech = sum(s["end"] - s["start"] for s in stamps)
        return {"speech_ratio": round(speech / max(1, len(audio)), 3),
                "speech_at_start": bool(stamps) and stamps[0]["start"] < self.edge,
                "speech_at_end": bool(stamps) and stamps[-1]["end"] > len(audio) - self.edge}


def cut_chunks(display_words: list[str], ref_tokens: list[str], ref_origin: list[int],
               asr_words: list[dict[str, Any]], wav: Path, gaps: list[tuple[float, float]],
               config: DataPrepConfig, speech_checker: SpeechChecker | None = None
               ) -> tuple[list[dict], list[dict], dict[str, Any]]:
    from rapidfuzz.distance import Levenshtein

    hyp = hyp_tokens(asr_words)
    opcodes = Levenshtein.opcodes(ref_tokens, [t[0] for t in hyp])
    matched = sum(r1 - r0 for tag, r0, r1, _, _ in opcodes if tag == "equal")
    bounds = find_boundaries(ref_origin, hyp, opcodes, gaps, config)
    stats = {"reference_tokens": len(ref_tokens), "asr_tokens": len(hyp),
             "matched_reference_fraction": round(matched / max(1, len(ref_tokens)), 3),
             "cut_points": len(bounds)}
    kept: list[dict] = []
    rejected: list[dict] = []
    sr = config.sample_rate
    i = 0
    while i < len(bounds) - 1:
        left, best = bounds[i], None
        for j in range(i + 1, len(bounds)):
            duration = bounds[j].left_end - left.right_start
            if duration > config.max_seconds:
                break
            if duration >= config.min_seconds:
                best = j
                if duration >= config.target_seconds:
                    break
        if best is None:  # no cut point at an acceptable distance; move on
            i += 1
            continue
        right = bounds[best]
        start, end = left.right_start, right.left_end
        ref_chunk = ref_tokens[left.ref:right.ref]
        hyp_chunk = [t[0] for t in hyp[left.hyp:right.hyp]]
        ref_norm, hyp_norm = " ".join(ref_chunk), " ".join(hyp_chunk)
        text = " ".join(display_words[ref_origin[left.ref]:ref_origin[right.ref - 1] + 1])
        cer = Levenshtein.distance(ref_norm, hyp_norm) / max(1, len(ref_norm))
        mismatch = longest_mismatch(ref_chunk, hyp_chunk)
        rate = len(ref_chunk) / (end - start)
        chunk = {"start": round(start, 3), "end": round(end, 3), "text": text, "text_norm": ref_norm,
                 "asr_norm": hyp_norm, "cer": round(cer, 4), "mismatch_run": mismatch,
                 "words_per_second": round(rate, 2)}
        reason = None
        if mismatch >= config.max_mismatch_run:
            reason = "mismatch_run"
        elif cer > config.max_cer:
            reason = "cer"
        elif not config.min_words_per_second <= rate <= config.max_words_per_second:
            reason = "speech_rate"
        elif speech_checker is not None:
            audio, _ = sf.read(wav, start=int(start * sr), stop=int(end * sr), dtype="float32")
            vad = speech_checker(audio)
            chunk.update(vad)
            if vad["speech_ratio"] < config.min_speech_ratio:
                reason = "low_speech_ratio"
            elif vad["speech_at_start"] or vad["speech_at_end"]:
                reason = "speech_at_edge"
        if reason:
            rejected.append({**chunk, "reason": reason})
        else:
            kept.append(chunk)
        i = best
    return kept, rejected, stats

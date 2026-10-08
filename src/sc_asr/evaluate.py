"""Generate hypotheses and calculate held-out corpus/per-example WER and CER."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .metrics import clean_text, score_pairs


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", help="LoRA adapter directory; omit to score the base model as-is")
    parser.add_argument("--base-model", default="openai/whisper-tiny")
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="en")


def max_tokens(seconds: float) -> int:
    """Generous cap on output length: fast speech is ~5 tokens/s, so 8/s + 10 never cuts real text."""
    return min(440, int(seconds * 8) + 10)


def compression_ratio(text: str) -> float:
    """Whisper's own loop detector: repetitive output compresses far better than speech (> 2.4)."""
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


def evaluate(args: argparse.Namespace) -> dict:
    manifest = Path(args.manifest).resolve()
    items = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise ValueError("Evaluation manifest is empty")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model or args.base_model,
                                                 language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    if args.model:
        model = PeftModel.from_pretrained(model, args.model)
    model = model.to(device).eval()
    references, hypotheses, details = [], [], []
    for item in items:
        audio_path = Path(item["audio"])
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"{audio_path}: expected 16 kHz, found {sr}")
        inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            tokens = model.generate(inputs, language=args.language, task="transcribe",
                                    max_new_tokens=max_tokens(len(audio) / sr))
        hypothesis = clean_text(processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
        reference = clean_text(item["text"])
        references.append(reference)
        hypotheses.append(hypothesis)
        details.append({"hearing_id": item.get("hearing_id"), "source_id": item.get("source_id"),
                        "reference": reference, "hypothesis": hypothesis,
                        "compression_ratio": round(compression_ratio(hypothesis), 2),
                        **score_pairs([reference], [hypothesis])})
    result = {"model": args.model or "zero-shot", "base_model": args.base_model, "examples": len(items),
              "repetition_loops": sum(d["compression_ratio"] > 2.4 for d in details),
              **score_pairs(references, hypotheses), "per_example": details}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


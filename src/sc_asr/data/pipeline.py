"""Staged runner for `sc-asr data-prep`. Every stage skips work whose output already exists.

  0 audio        pair audio with transcript PDFs, decode audio to 16 kHz mono WAV   -> wav16k/
  1 transcripts  extract and clean PDF text                                          -> text/
  2 anchors      Silero VAD speech map; faster-whisper word timestamps in windows    -> vad/, asr/
  3 align        align reference to anchors, cut at VAD silences, verify chunks     -> segments/
  4 export       case-level split, chunk WAVs, NeMo + HF manifests, review sample    -> chunks/, manifests/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import DataPrepConfig, load_config
from .pairing import Hearing, decode_to_wav, find_hearings

STAGES = ("audio", "transcripts", "anchors", "align", "export")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_audio(config: DataPrepConfig, hearings: dict[str, Hearing]) -> None:
    for key, hearing in hearings.items():
        target = config.out / "wav16k" / f"{key}.wav"
        if not target.exists():
            print(f"[audio] {hearing.audio.name} -> {target.name}", flush=True)
            decode_to_wav(hearing.audio, target, config.sample_rate)


def stage_transcripts(config: DataPrepConfig, hearings: dict[str, Hearing]) -> None:
    from .transcripts import transcript_text

    report_path = config.out / "reports" / "transcripts.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for key, hearing in hearings.items():
        target = config.out / "text" / f"{key}.txt"
        if target.exists():
            continue
        print(f"[transcripts] {hearing.transcript.name}", flush=True)
        text, info = transcript_text(hearing.transcript, config.line_numbers)
        if not info["start_marker_found"]:
            print(f"  warning: no start marker in {hearing.transcript.name}; kept all text", flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        report[key] = info
    _write_json(report_path, report)


def _speech_map(config: DataPrepConfig, key: str) -> dict[str, Any]:
    from .vad import load_speech_map

    return load_speech_map(config.out / "wav16k" / f"{key}.wav", config.out / "vad" / f"{key}.json")


def stage_anchors(config: DataPrepConfig, hearings: dict[str, Hearing]) -> None:
    from .anchors import WordTimestamper
    from .vad import silences

    timestamper = None
    for key in hearings:
        target = config.out / "asr" / f"{key}.json"
        if target.exists():
            continue
        speech_map = _speech_map(config, key)
        if timestamper is None:
            timestamper = WordTimestamper(config.asr_model, config.device, config.compute_type, config.beam_size,
                                          config.cpu_threads, config.window_minutes)
        print(f"[anchors] {key}", flush=True)
        _write_json(target, timestamper(config.out / "wav16k" / f"{key}.wav", silences(speech_map),
                                        speech_map["duration"]))


def stage_align(config: DataPrepConfig, hearings: dict[str, Hearing]) -> None:
    from .align import SpeechChecker, cut_chunks
    from .textnorm import tokenize_reference
    from .vad import silences

    checker = SpeechChecker(config.sample_rate, config.edge_window) if config.vad_check else None
    out = config.out
    (out / "segments").mkdir(parents=True, exist_ok=True)
    report = {}
    for key in hearings:
        words, tokens, origin = tokenize_reference((out / "text" / f"{key}.txt").read_text(encoding="utf-8"))
        asr_words = json.loads((out / "asr" / f"{key}.json").read_text(encoding="utf-8"))
        gaps = silences(_speech_map(config, key))
        kept, rejected, stats = cut_chunks(words, tokens, origin, asr_words, out / "wav16k" / f"{key}.wav",
                                           gaps, config, checker)
        _write_jsonl(out / "segments" / f"{key}.jsonl", kept)
        _write_jsonl(out / "segments" / f"{key}.rejected.jsonl", rejected)
        report[key] = {**stats, "kept": len(kept),
                       "kept_hours": round(sum(c["end"] - c["start"] for c in kept) / 3600, 3),
                       "rejected": dict(Counter(c["reason"] for c in rejected))}
        print(f"[align] {key} {report[key]}", flush=True)
    _write_json(out / "reports" / "alignment.json", report)


def run(config: DataPrepConfig, first: int, last: int) -> dict[str, Any]:
    hearings, pairing = find_hearings(Path(config.transcripts_dir), [Path(d) for d in config.audio_dirs])
    if config.hearings:
        unknown = sorted(set(config.hearings) - set(hearings))
        if unknown:
            raise ValueError(f"Hearings in config not found among paired hearings: {unknown}")
        hearings = {k: hearings[k] for k in config.hearings}
        pairing["selected"] = list(config.hearings)
    _write_json(config.out / "reports" / "pairing.json", pairing)
    print(f"{len(hearings)} hearings paired; {len(pairing['not_audio'])} files skipped as not audio "
          f"(see {config.out / 'reports' / 'pairing.json'})", flush=True)
    if not hearings:
        raise RuntimeError("No audio/transcript pairs found. Check audio_dirs in the config; files under "
                           "the audio directories must be real audio, not saved web pages.")
    steps = [stage_audio, stage_transcripts, stage_anchors, stage_align]
    result: dict[str, Any] = {"hearings": len(hearings)}
    for index in range(first, last + 1):
        print(f"===== stage {index}: {STAGES[index]} =====", flush=True)
        if index < len(steps):
            steps[index](config, hearings)
        else:
            from .export import export

            result["splits"] = export(config, hearings)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/data_prep.yaml", help="YAML settings file")
    parser.add_argument("--stage", type=int, default=0, choices=range(len(STAGES)),
                        help="First stage: " + ", ".join(f"{i}={s}" for i, s in enumerate(STAGES)))
    parser.add_argument("--stop-stage", type=int, default=len(STAGES) - 1, choices=range(len(STAGES)))
    parser.add_argument("--output-dir", help="Override output_dir from the config")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Override device from the config")
    parser.add_argument("--asr-model", help="Override asr_model from the config")


def main(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage > args.stop_stage:
        raise ValueError("--stage must not be greater than --stop-stage")
    config = load_config(args.config, {"output_dir": args.output_dir, "device": args.device,
                                       "asr_model": args.asr_model})
    return run(config, args.stage, args.stop_stage)

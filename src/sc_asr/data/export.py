"""Case-level split and export of kept chunks to NeMo and Whisper (HF JSONL) manifests."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

from .config import DataPrepConfig
from .pairing import Hearing

SPLITS = ("train", "validation", "test")


def assign_splits(hours_by_case: dict[str, float], config: DataPrepConfig) -> dict[str, str]:
    """Cases pinned in config.case_splits keep their split. The rest are shuffled with a fixed seed and
    fill test, then validation, up to their share of the audio; everything else is train."""
    splits = {c: s for c, s in config.case_splits.items() if c in hours_by_case}
    total = sum(hours_by_case.values())
    targets = {"test": config.test_fraction * total, "validation": config.validation_fraction * total}
    filled = {name: sum(hours_by_case[c] for c, s in splits.items() if s == name) for name in targets}
    cases = sorted(set(hours_by_case) - set(splits))
    random.Random(config.seed).shuffle(cases)
    for case in cases:
        for name in ("test", "validation"):
            if filled[name] < targets[name]:
                splits[case] = name
                filled[name] += hours_by_case[case]
                break
        else:
            splits[case] = "train"
    return splits


def _relative(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def export(config: DataPrepConfig, hearings: dict[str, Hearing]) -> dict[str, Any]:
    out = config.out
    chunks: dict[str, list[dict]] = {}
    for key in hearings:
        path = out / "segments" / f"{key}.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                chunks[key] = rows
    if not chunks:
        raise RuntimeError("No kept chunks found; run the align stage first and check reports/alignment.json")
    hours_by_case: dict[str, float] = defaultdict(float)
    for key, rows in chunks.items():
        hours_by_case[hearings[key].case_id] += sum(r["end"] - r["start"] for r in rows) / 3600
    case_split = assign_splits(hours_by_case, config)

    chunk_root = out / "chunks"
    if chunk_root.exists():  # generated output; rebuilt so a changed split leaves no stale files
        shutil.rmtree(chunk_root)
    nemo_dir, hf_dir = out / "manifests" / "nemo", out / "manifests" / "hf"
    nemo_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    for split in SPLITS:
        handles[split] = (open(nemo_dir / f"{split}.json", "w", encoding="utf-8", newline="\n"),
                          open(hf_dir / f"{split}.jsonl", "w", encoding="utf-8", newline="\n"))
    summary: dict[str, dict[str, Any]] = {s: {"cases": set(), "hearings": 0, "utterances": 0, "hours": 0.0}
                                          for s in SPLITS}
    review_pool = []
    try:
        for key in sorted(chunks):
            hearing = hearings[key]
            split = case_split[hearing.case_id]
            wav = out / "wav16k" / f"{key}.wav"
            split_dir = chunk_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(wav) as source:
                sr = source.samplerate
                for index, row in enumerate(chunks[key]):
                    utt_id = f"{key}-{index:05d}"
                    source.seek(int(row["start"] * sr))
                    audio = source.read(int(row["end"] * sr) - int(row["start"] * sr), dtype="int16")
                    chunk_path = split_dir / f"{utt_id}.wav"
                    sf.write(chunk_path, audio, sr, subtype="PCM_16")
                    duration = round(len(audio) / sr, 3)
                    nemo, hf = handles[split]
                    nemo.write(json.dumps({
                        "audio_filepath": _relative(chunk_path, nemo_dir), "duration": duration,
                        "text": row["text"], "text_norm": row["text_norm"], "utt_id": utt_id,
                        "hearing_id": key, "case_id": hearing.case_id, "start": row["start"],
                        "end": row["end"], "cer": row["cer"]}, ensure_ascii=False) + "\n")
                    hf.write(json.dumps({
                        "audio": _relative(chunk_path, hf_dir), "text": row["text"], "hearing_id": key,
                        "source_id": utt_id, "start_sec": row["start"], "end_sec": row["end"]},
                        ensure_ascii=False) + "\n")
                    review_pool.append({"utt_id": utt_id, "split": split,
                                        "audio": _relative(chunk_path, out), "duration": duration,
                                        "cer": row["cer"], "text": row["text"]})
                    stats = summary[split]
                    stats["utterances"] += 1
                    stats["hours"] += duration / 3600
            summary[split]["cases"].add(hearing.case_id)
            summary[split]["hearings"] += 1
    finally:
        for nemo, hf in handles.values():
            nemo.close()
            hf.close()

    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    sample = random.Random(config.seed).sample(review_pool, min(config.review_sample_size, len(review_pool)))
    with open(reports / "review_sample.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "split", "audio", "duration", "cer", "text"])
        writer.writeheader()
        writer.writerows(sample)
    result = {split: {**s, "cases": sorted(s["cases"]), "hours": round(s["hours"], 2)}
              for split, s in summary.items()}
    (reports / "splits.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

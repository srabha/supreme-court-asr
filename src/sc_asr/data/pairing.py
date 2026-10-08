"""Match audio files to transcript PDFs and decode audio to 16 kHz mono WAV."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIO_SUFFIXES = {".mp3", ".m4a", ".mp4", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm"}


@dataclass(frozen=True)
class Hearing:
    key: str          # e.g. "161132009_2023-12-12"
    case_id: str      # e.g. "161132009"; hearings of one case never cross splits
    audio: Path
    transcript: Path


def hearing_key(stem: str) -> str:
    """"001_161132009_2023-12-12" -> "161132009_2023-12-12" (drops the download index)."""
    return re.sub(r"^\d{3}_", "", stem)


def case_id(key: str) -> str:
    return key.split("_")[0]


def is_audio(path: Path) -> bool:
    """Sniff the file header; download failures are often saved HTML pages with an audio extension."""
    with path.open("rb") as f:
        head = f.read(16)
    if len(head) < 12:
        return False
    return (head[:3] == b"ID3" or (head[0] == 0xFF and head[1] & 0xE0 == 0xE0) or head[4:8] == b"ftyp"
            or head[:4] in (b"RIFF", b"fLaC", b"OggS") or head[:4] == b"\x1aE\xdf\xa3")


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_hearings(transcripts_dir: Path, audio_dirs: list[Path]) -> tuple[dict[str, Hearing], dict[str, Any]]:
    report: dict[str, Any] = {"not_audio": [], "duplicate_transcripts": [], "missing_dirs": []}
    pdfs: dict[str, Path] = {}
    seen_hashes: dict[str, Path] = {}
    if not transcripts_dir.is_dir():
        report["missing_dirs"].append(str(transcripts_dir))
    for pdf in sorted(transcripts_dir.glob("*.pdf")):
        digest = _sha1(pdf)
        if digest in seen_hashes:
            report["duplicate_transcripts"].append({"file": str(pdf), "same_as": str(seen_hashes[digest])})
            continue
        seen_hashes[digest] = pdf
        pdfs[hearing_key(pdf.stem)] = pdf
    audio: dict[str, Path] = {}
    for directory in audio_dirs:
        if not directory.is_dir():
            report["missing_dirs"].append(str(directory))
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            if not is_audio(path):
                report["not_audio"].append(str(path))
                continue
            audio.setdefault(hearing_key(path.stem), path)
    hearings = {k: Hearing(k, case_id(k), audio[k], pdfs[k]) for k in sorted(audio.keys() & pdfs.keys())}
    report.update({
        "paired": sorted(hearings),
        "transcript_without_audio": sorted(pdfs.keys() - audio.keys()),
        "audio_without_transcript": sorted(audio.keys() - pdfs.keys()),
    })
    return hearings, report


def decode_to_wav(source: Path, target: Path, sample_rate: int = 16000) -> float:
    """Decode any FFmpeg-readable file to mono 16-bit PCM WAV; returns duration in seconds."""
    import av
    import soundfile as sf

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.wav")
    samples = 0
    with av.open(str(source)) as container, sf.SoundFile(partial, "w", sample_rate, 1, "PCM_16") as out:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(container.streams.audio[0]):
            for chunk in resampler.resample(frame):
                data = chunk.to_ndarray().reshape(-1)
                out.write(data)
                samples += len(data)
        for chunk in resampler.resample(None):  # flush buffered samples
            data = chunk.to_ndarray().reshape(-1)
            out.write(data)
            samples += len(data)
    partial.replace(target)
    return samples / sample_rate

"""Data-prep settings, loaded from a YAML file with optional overrides."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DataPrepConfig:
    transcripts_dir: str = "sc_hearings/transcripts"
    audio_dirs: list[str] = field(default_factory=lambda: ["sc_hearings/mp3"])
    output_dir: str = "data/prepared"
    hearings: list[str] = field(default_factory=list)       # empty = all paired hearings
    case_splits: dict[str, str] = field(default_factory=dict)  # pin cases to train/validation/test
    sample_rate: int = 16000
    line_numbers: str = "trailing"
    asr_model: str = "medium.en"
    device: str = "cuda"
    compute_type: str | None = None
    beam_size: int = 5
    cpu_threads: int = 4
    window_minutes: float = 20.0
    min_anchor_words: int = 3
    min_pause: float = 0.15
    snap_slack: float = 0.2
    edge_pad: float = 0.15
    min_seconds: float = 2.0
    target_seconds: float = 10.0
    max_seconds: float = 20.0
    max_mismatch_run: int = 4
    max_cer: float = 0.35
    min_words_per_second: float = 0.8
    max_words_per_second: float = 5.5
    vad_check: bool = True
    min_speech_ratio: float = 0.6
    edge_window: float = 0.08
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 13
    review_sample_size: int = 50

    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    def validate(self) -> None:
        if not 0 < self.min_seconds <= self.target_seconds <= self.max_seconds:
            raise ValueError("Require 0 < min_seconds <= target_seconds <= max_seconds")
        if self.validation_fraction < 0 or self.test_fraction < 0 or \
                self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("Validation and test fractions must be nonnegative and sum to less than one")
        bad_splits = {c: s for c, s in self.case_splits.items() if s not in {"train", "validation", "test"}}
        if bad_splits:
            raise ValueError(f"case_splits values must be train, validation or test: {bad_splits}")
        if self.line_numbers not in {"trailing", "leading", "none"}:
            raise ValueError("line_numbers must be 'trailing', 'leading' or 'none'")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")


def load_config(path: str | None, overrides: dict[str, Any] | None = None) -> DataPrepConfig:
    values: dict[str, Any] = {}
    if path:
        import yaml

        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    known = {f.name for f in fields(DataPrepConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown data-prep settings: {unknown}")
    values = {k: v for k, v in values.items() if v is not None}  # "hearings:" with no items means all
    values["case_splits"] = {str(k): v for k, v in (values.get("case_splits") or {}).items()}
    config = DataPrepConfig(**values)
    config.validate()
    return config

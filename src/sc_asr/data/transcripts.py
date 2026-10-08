"""Extract and clean reference text from hearing transcript PDFs.

Layout handled (Supreme Court of India transcripts, "Transcribed by TERES"): a cover page and a
document-control block, then the proceedings starting at a clock time ("10:45 AM IST"), speaker labels
("CHIEF JUSTICE DY CHANDRACHUD:"), a line number at the end of every line, a running page header, and
an "END OF THIS PROCEEDING" marker.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

# "CHIEF JUSTICE D.Y. CHANDRACHUD:" / "MR. KAPIL SIBAL:" at the start of a line
SPEAKER_RE = re.compile(r"^(?:[A-Z][A-Z.'’()\-]*\s?){1,8}:\s*")
# Transcriber notes that are not speech
NOTE_RE = re.compile(r"\[[^\]]*\]|\((?:inaudible|crosstalk|unclear|indistinct|laughter|pause|interruption)[^)]*\)",
                     re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"[\d\s.\-–|]+|page\s+\d+(\s+of\s+\d+)?", re.IGNORECASE)
CLOCK_RE = re.compile(r"\d{1,2}:\d{2}\s*(?:AM|PM)(?:\s*IST)?", re.IGNORECASE)
END_RE = re.compile(r"END OF (?:THIS |THE |DAY['’]S )?PROCEEDINGS?\b", re.IGNORECASE)
LINE_NUMBER_RE = {"trailing": re.compile(r"\s+\d{1,2}$"), "leading": re.compile(r"^\d{1,2}\s+")}


def extract_pages(pdf: Path) -> list[list[str]]:
    from pypdf import PdfReader

    return [(page.extract_text() or "").splitlines() for page in PdfReader(pdf).pages]


def clean_pages(pages: list[list[str]], line_numbers: str = "trailing") -> tuple[str, dict[str, Any]]:
    """Keep only the spoken proceedings; returns (text, info about what was found)."""
    if line_numbers not in ("trailing", "leading", "none"):
        raise ValueError("line_numbers must be 'trailing', 'leading' or 'none'")
    # A line repeated on many pages is a running header or footer.
    counts = Counter(line.strip() for page in pages for line in set(page) if line.strip())
    boilerplate = {line for line, n in counts.items() if n >= max(3, 0.3 * len(pages))}
    lines = []
    for page in pages:
        for line in page:
            text = line.strip()
            if not text or text in boilerplate or PAGE_NUMBER_RE.fullmatch(text):
                continue
            if line_numbers != "none":
                text = LINE_NUMBER_RE[line_numbers].sub("", text).strip()
            if text:
                lines.append(text)

    # Proceedings start at the first clock time or speaker label; everything before is the cover page.
    start = next((i for i, t in enumerate(lines) if CLOCK_RE.fullmatch(t) or SPEAKER_RE.match(t)), None)
    end = next((i for i, t in enumerate(lines) if END_RE.match(t)), None)
    info = {"start_marker_found": start is not None, "end_marker_found": end is not None,
            "lines_dropped_before_start": start or 0}
    kept = []
    for text in lines[start or 0:end]:
        if CLOCK_RE.fullmatch(text):
            continue
        kept.append(SPEAKER_RE.sub("", text))
    text = NOTE_RE.sub(" ", " ".join(kept))
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # words hyphenated across a line break
    text = re.sub(r"\s+", " ", text).strip()
    info["words"] = len(text.split())
    return text, info


def transcript_text(pdf: Path, line_numbers: str = "trailing") -> tuple[str, dict[str, Any]]:
    return clean_pages(extract_pages(pdf), line_numbers)

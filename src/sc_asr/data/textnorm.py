"""Two views of transcript text.

* display text: the transcript as written (case, punctuation, digits) -> training label for Parakeet/Whisper.
* normalised tokens: lowercase words, numbers spelled out, no punctuation -> used only to match the
  reference against ASR output and to score it. The same function is applied to both sides.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", " ": " "})
_SYMBOLS = {"&": "and", "%": "percent", "§": "section", "@": "at", "+": "plus", "₹": "rupees"}
_WORD_MAP = {"mr": "mister", "mrs": "missus", "dr": "doctor", "vs": "versus", "hon'ble": "honourable",
             "honble": "honourable"}


def clean_display(text: str) -> str:
    """Unicode/whitespace clean-up only; keeps meaning, case and punctuation."""
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


def _cardinal(digits: str) -> str:
    from num2words import num2words

    if len(digits) > 12 or (len(digits) > 1 and digits.startswith("0")):  # ids, phone numbers, "007"
        return " ".join(num2words(int(d)) for d in digits)
    return num2words(int(digits))


def normalize_word(word: str) -> list[str]:
    """Normalise one written word into zero or more lowercase tokens."""
    from num2words import num2words

    text = unicodedata.normalize("NFKC", word).translate(_QUOTES).lower()
    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, f" {spoken} ")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)                                   # 1,000 -> 1000
    text = re.sub(r"(\d+)\.(\d+)",
                  lambda m: f" {_cardinal(m[1])} point {' '.join(_cardinal(d) for d in m[2])} ", text)
    text = re.sub(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", lambda m: f" {num2words(int(m[1]), to='year')} ", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: f" {num2words(int(m[1]), to='ordinal')} ", text)
    text = re.sub(r"\d+", lambda m: f" {_cardinal(m[0])} ", text)                  # 6a -> six a
    text = re.sub(r"[^a-z']+", " ", text)
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)                              # keep don't, drop quotes
    return [_WORD_MAP.get(token, token) for token in text.split()]


def normalize_text(text: str) -> str:
    return " ".join(token for word in text.split() for token in normalize_word(word))


def tokenize_reference(text: str) -> tuple[list[str], list[str], list[int]]:
    """Return (display words, normalised tokens, index of the display word each token came from)."""
    words = clean_display(text).split()
    tokens: list[str] = []
    origin: list[int] = []
    for index, word in enumerate(words):
        for token in normalize_word(word):
            tokens.append(token)
            origin.append(index)
    return words, tokens, origin

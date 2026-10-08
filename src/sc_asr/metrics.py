"""Text cleanup and ASR metrics."""

import re
import unicodedata

import jiwer


def clean_text(text: str) -> str:
    """Tidy Unicode, quote marks and whitespace; case, punctuation and digits are kept."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def metric_text(text: str) -> str:
    """Form used for scoring: lowercase, punctuation removed, apostrophes inside words kept."""
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_pairs(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Corpus-level WER and CER, plus the WER split into substitution, deletion and insertion rates.

    Each rate is a count divided by the number of reference words, so the three add up to the WER.
    """
    refs = [metric_text(s) for s in references]
    hyps = [metric_text(s) for s in hypotheses]
    words = jiwer.process_words(refs, hyps)
    n = max(1, words.hits + words.substitutions + words.deletions)
    return {"wer": float(words.wer), "cer": float(jiwer.cer(refs, hyps)),
            "substitutions": words.substitutions / n, "deletions": words.deletions / n,
            "insertions": words.insertions / n}

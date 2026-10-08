from rapidfuzz.distance import Levenshtein

from sc_asr.data.align import find_boundaries, longest_mismatch
from sc_asr.data.config import DataPrepConfig


def test_substitutions_are_not_mismatches():
    ref = "give a copy to the learned solicitor general".split()
    hyp = "give a copy to the learned solution".split()
    assert longest_mismatch(ref, hyp) == 1


def test_skipped_passage_is_a_mismatch():
    ref = "my lord no no no do not i am requesting the court an application".split()
    hyp = "my lord no no no an application".split()
    assert longest_mismatch(ref, hyp) == 7


def test_cuts_snap_to_vad_silence_and_need_one():
    words = "one two three four five six".split()
    # pause between "three" and "four" (1.5 -> 1.8 s), no pause elsewhere
    hyp = [("one", 0.0, 0.5), ("two", 0.5, 1.0), ("three", 1.0, 1.5),
           ("four", 1.8, 2.3), ("five", 2.3, 2.8), ("six", 2.8, 3.3)]
    ops = Levenshtein.opcodes(words, [h[0] for h in hyp])
    origin = list(range(len(words)))
    config = DataPrepConfig()

    bounds = find_boundaries(origin, hyp, ops, [(1.55, 1.9)], config)
    assert len(bounds) == 1
    cut = bounds[0]
    assert (cut.ref, cut.hyp) == (3, 3)
    assert 1.55 < cut.left_end <= cut.right_start < 1.9

    assert find_boundaries(origin, hyp, ops, [], config) == []  # ASR pause without real silence

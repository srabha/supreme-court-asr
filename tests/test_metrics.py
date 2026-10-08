from pytest import approx

from sc_asr.metrics import clean_text, score_pairs


def test_case_and_punctuation_are_ignored():
    scores = score_pairs(["My Lord, the Section applies."], ["my lord the section applies"])
    assert scores["wer"] == 0.0 and scores["cer"] == 0.0


def test_word_errors_are_counted_over_the_corpus():
    # one substitution and one deletion in eight reference words
    scores = score_pairs(["the learned solicitor general", "may it please you"],
                         ["the learned solution general", "may it please"])
    assert scores["wer"] == 0.25
    assert scores["substitutions"] == scores["deletions"] == 0.125
    assert scores["insertions"] == 0.0


def test_insertions_can_push_wer_above_one():
    scores = score_pairs(["my lord"], ["my lord my lord my lord"])
    assert scores["wer"] == 2.0
    assert scores["insertions"] == 2.0


def test_error_rates_add_up_to_wer():
    scores = score_pairs(["the bench will hear the matter today"], ["a bench hear the the matter today now"])
    assert scores["substitutions"] + scores["deletions"] + scores["insertions"] == approx(scores["wer"])


def test_apostrophes_inside_words_are_kept():
    assert score_pairs(["they've"], ["they ve"])["wer"] > 0
    assert clean_text("it’s  fine\n") == "it's fine"

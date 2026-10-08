from sc_asr.data.textnorm import normalize_text, tokenize_reference


def test_numbers_and_symbols():
    assert normalize_text("Section 6A, 1,000 vehicles in 2023") == \
        "section six a one thousand vehicles in twenty twenty three"
    assert normalize_text("the 1st & 2nd respondents, 7.5%") == \
        "the first and second respondents seven point five percent"


def test_abbreviations_and_apostrophes():
    assert normalize_text("Hon'ble Mr. Justice") == "honourable mister justice"
    assert normalize_text("Don't say 'no'") == "don't say no"


def test_tokens_point_back_to_written_words():
    words, tokens, origin = tokenize_reference("Paragraph 11 is set out.")
    assert words == ["Paragraph", "11", "is", "set", "out."]
    assert tokens == ["paragraph", "eleven", "is", "set", "out"]
    assert origin == [0, 1, 2, 3, 4]

    words, tokens, origin = tokenize_reference("about 1,000 cases")
    assert tokens == ["about", "one", "thousand", "cases"]
    assert origin == [0, 1, 1, 2]

from sc_asr.data.transcripts import clean_pages

HEADER = "Transcribed by TERES"


def make_pages():
    cover = [HEADER, "1", "CHIEF JUSTICE'S COURT", "Civil Appeal No.841/2018", "Document Control",
             "Version 1.1"]
    page2 = [HEADER, "2", "10:45 AM IST",
             "ATTORNEY GENERAL: There's a note...   1",
             " 2 ",
             "CHIEF JUSTICE: Yes, let's have a quick look at  3",
             "the note now.  4"]
    page3 = [HEADER, "3", "MR. SIBAL: Paragraph  5", "11 is set out there.  6",
             "END OF DAY’S PROCEEDINGS 7"]
    return [cover, page2, page3]


def test_keeps_only_spoken_text():
    text, info = clean_pages(make_pages())
    assert text == ("There's a note... Yes, let's have a quick look at the note now. "
                    "Paragraph 11 is set out there.")
    assert info["start_marker_found"] and info["end_marker_found"]


def test_missing_markers_keep_everything():
    pages = [["Some text 1", "more text 2"], ["and a bit 3"], ["more at the end 4"]]
    text, info = clean_pages(pages)
    assert not info["start_marker_found"]
    assert text == "Some text more text and a bit more at the end"

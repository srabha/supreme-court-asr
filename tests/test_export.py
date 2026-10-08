from sc_asr.data.config import DataPrepConfig
from sc_asr.data.export import assign_splits

HOURS = {"a": 1.3, "b": 1.3, "c": 0.4, "d": 0.5, "e": 0.9}


def test_pinned_cases_keep_their_split():
    config = DataPrepConfig(case_splits={"d": "validation", "e": "test"})
    splits = assign_splits(HOURS, config)
    assert splits["d"] == "validation" and splits["e"] == "test"
    assert {splits[c] for c in "abc"} == {"train"}  # held-out targets already met


def test_every_case_in_exactly_one_split():
    splits = assign_splits(HOURS, DataPrepConfig())
    assert set(splits) == set(HOURS)
    assert set(splits.values()) <= {"train", "validation", "test"}

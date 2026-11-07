"""unit tests for the pure mixing logic in pipelines/build_corpora.py (no network, no tokenizer)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
from build_corpora import balance_merge, build_shared, length_matched_extras, render_texts


def decode(ids: list[int]) -> str:
    return " ".join(str(i) for i in ids)


def test_render_texts_uses_text_field():
    rows = [{"content": "doc one"}, {"content": "doc two"}]
    assert render_texts(rows, {"text_field": "content"}) == ["doc one", "doc two"]


def test_render_texts_formats_template():
    rows = [{"user_prompt": "q?", "assistant_response": "a.", "extra": "ignored"}]
    src = {"text_template": "{user_prompt}\n\n{assistant_response}"}
    assert render_texts(rows, src) == ["q?\n\na."]


def test_render_texts_template_missing_field_raises():
    with pytest.raises(KeyError):
        render_texts([{"user_prompt": "q?"}], {"text_template": "{assistant_response}"})


def test_balance_merge_keeps_order_and_covers_all():
    a, at = ["a0", "a1"], [5, 7]
    b, bt = ["b0", "b1", "b2"], [4, 4, 4]
    out = balance_merge(a, at, b, bt)
    assert [d for t, d in out if t == "a"] == a
    assert [d for t, d in out if t == "b"] == b
    assert len(out) == len(a) + len(b)


def test_balance_merge_alternates_by_token_balance():
    out = balance_merge(["a0"], [10], ["b0", "b1"], [5, 5])
    assert [t for t, _ in out] == ["a", "b", "b"]


def test_build_shared_truncates_and_hits_target():
    replay = [[1] * 10, [2] * 10, [3] * 10]
    texts, toks, cs, used = build_shared(replay, target=12, cap=8, decode=decode)
    assert toks == [8, 8]
    assert cs == 16
    assert used == 2
    assert texts[0] == decode([1] * 8)


def test_build_shared_exhaustion_asserts():
    with pytest.raises(AssertionError):
        build_shared([[1] * 4], target=100, cap=8, decode=decode)


def test_length_matched_extras_matches_lengths_exactly():
    replay = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    lengths = [2, 4, 3]
    extras, used = length_matched_extras(replay, 0, lengths, decode)
    assert [n for _, n in extras] == lengths
    assert used == len(replay)
    assert extras[0][0] == decode([1, 2])
    assert extras[1][0] == decode([3, 4, 5, 6])


def test_length_matched_extras_exhaustion_asserts():
    with pytest.raises(AssertionError):
        length_matched_extras([[1]], 0, [5], decode)


def test_control_interleave_matches_reference_arm():
    ref_docs, ref_toks = ["r0", "r1", "r2"], [3, 6, 2]
    shared_texts, shared_toks = ["s0", "s1", "s2"], [4, 4, 4]
    replay = [[9] * 7, [8] * 7]
    extras, _ = length_matched_extras(replay, 0, ref_toks, decode)
    mix = balance_merge(ref_docs, ref_toks, shared_texts, shared_toks)
    ctrl = balance_merge([t for t, _ in extras], [n for _, n in extras], shared_texts, shared_toks)
    assert [t for t, _ in mix] == [t for t, _ in ctrl]
    assert [d for t, d in mix if t == "b"] == [d for t, d in ctrl if t == "b"]

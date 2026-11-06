"""Regression tests for token counter cache configuration."""

import pytest

import semchunk


@pytest.mark.parametrize("sizes", [(None, 2), (2, None), (2, 0), (0, 2), (2, 3)])
@pytest.mark.parametrize("direct_first", [False, True])
def test_cache_maxsize_respected(sizes, direct_first):
    def count_words(text):
        return len(text.split())

    first_size, second_size = sizes
    if direct_first:
        semchunk.chunk("one two three", 2, count_words, cache_maxsize=first_size)
    else:
        semchunk.chunkerify(count_words, 2, cache_maxsize=first_size)("one two three")

    chunker = semchunk.chunkerify(count_words, 2, cache_maxsize=second_size)
    for number in range(10):
        text = f"example number {number}"
        chunks, offsets = chunker(text, offsets=True)
        assert chunks == [text[start:end] for start, end in offsets]
        assert chunks == semchunk.chunk(text, 2, count_words, memoize=False)

    info = chunker.token_counter.cache_info()
    assert info.maxsize == second_size
    if second_size is not None:
        assert info.currsize <= second_size


def test_direct_chunk_uses_requested_cache():
    calls = []

    def count_words(text):
        calls.append(text)
        return len(text.split())

    semchunk.chunkerify(count_words, 2)("one two")
    calls.clear()
    assert semchunk.chunk("one two", 2, count_words, cache_maxsize=0) == ["one two"]
    assert "one two" in calls


def test_matching_cache_configuration_is_shared():
    def count_words(text):
        return len(text.split())

    first = semchunk.chunkerify(count_words, 2, cache_maxsize=2)
    second = semchunk.chunkerify(count_words, 2, cache_maxsize=2)
    assert first.token_counter is second.token_counter
    first("one two")
    before = first.token_counter.cache_info().hits
    semchunk.chunk("one two", 2, count_words, cache_maxsize=2)
    assert first.token_counter.cache_info().hits > before


def test_memoize_false_preserves_counter():
    def count_words(text):
        return len(text.split())

    semchunk.chunkerify(count_words, 2, cache_maxsize=2)
    chunker = semchunk.chunkerify(count_words, 2, cache_maxsize=0, memoize=False)
    assert chunker.token_counter is count_words
    assert chunker("one two") == ["one two"]

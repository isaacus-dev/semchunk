"""Check empty batch results and the neighboring input shapes."""

import pytest

import semchunk


@pytest.mark.parametrize("texts", [[], ()], ids=["list", "tuple"])
@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("progress", [False, True])
@pytest.mark.parametrize("processes", [1, 2])
def test_empty_batch(texts, offsets, progress, processes):
    chunker = semchunk.chunkerify(len, 4, memoize=False)

    assert chunker(texts, offsets=offsets, progress=progress, processes=processes) == (([], []) if offsets else [])


@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("progress", [False, True])
@pytest.mark.parametrize("processes", [1, 2])
def test_nonempty_batch(offsets, progress, processes):
    chunker = semchunk.chunkerify(len, 4, memoize=False)
    chunks = [["abcd", "ef"], []]
    expected = (chunks, [[(0, 4), (5, 7)], []]) if offsets else chunks

    assert chunker(("abcd ef", ""), offsets=offsets, progress=progress, processes=processes) == expected


@pytest.mark.parametrize("offsets", [False, True])
def test_empty_string(offsets):
    chunker = semchunk.chunkerify(len, 4, memoize=False)

    assert chunker("", offsets=offsets) == (([], []) if offsets else [])

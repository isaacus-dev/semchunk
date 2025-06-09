from __future__ import annotations

import re
import math
import inspect

from typing import Callable, Sequence, TYPE_CHECKING
from itertools import accumulate
from contextlib import suppress
from functools import lru_cache

import mpire

from tqdm import tqdm

if TYPE_CHECKING:
    import tiktoken
    import tokenizers
    import transformers


_memoized_token_counters = {}
"""A map of token counters to their memoized versions."""

_NON_WHITESPACE_SEMANTIC_SPLITTERS = (
    # Sentence terminators.
    ".",
    "?",
    "!",
    "*",
    # Clause separators.
    ";",
    ",",
    "(",
    ")",
    "[",
    "]",
    "“",
    "”",
    "‘",
    "’",
    "'",
    '"',
    "`",
    # Sentence interrupters.
    ":",
    "—",
    "…",
    # Word joiners.
    "/",
    "\\",
    "–",
    "&",
    "-",
)
"""A tuple of semantically meaningful non-whitespace splitters that may be used to chunk texts, ordered from most desirable to least desirable."""

_REGEX_ESCAPED_NON_WHITESPACE_SEMANTIC_SPLITTERS = tuple(re.escape(splitter) for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS)

def _split_text(text: str) -> tuple[str, bool, list[str]]:
    """Split text using the most semantically meaningful splitter possible."""

    splitter_is_whitespace = True

    # Try splitting at, in order of most desirable to least desirable:
    # - The largest sequence of newlines and/or carriage returns;
    # - The largest sequence of tabs;
    # - The largest sequence of whitespace characters or, if the largest such sequence is only a single character and there exists a whitespace character preceded by a semantically meaningful non-whitespace splitter, then that whitespace character;
    # - A semantically meaningful non-whitespace splitter.
    if "\n" in text or "\r" in text:
        splitter = max(re.findall(r"[\r\n]+", text))

    elif "\t" in text:
        splitter = max(re.findall(r"\t+", text))

    elif re.search(r"\s", text):
        splitter = max(re.findall(r"\s+", text))
        
        # If the splitter is only a single character, see if we can target whitespace characters that are preceded by semantically meaningful non-whitespace splitters to avoid splitting in the middle of sentences.
        if len(splitter) == 1:
            for escaped_preceder in _REGEX_ESCAPED_NON_WHITESPACE_SEMANTIC_SPLITTERS:
                if (whitespace_preceded_by_preceder := re.search(rf'{escaped_preceder}(\s)', text)):
                    splitter = whitespace_preceded_by_preceder.group(1)
                    escaped_splitter = re.escape(splitter)
                    
                    return splitter, splitter_is_whitespace, re.split(rf'(?<={escaped_preceder}){escaped_splitter}', text)

    else:
        # Identify the most desirable semantically meaningful non-whitespace splitter present in the text.
        for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS:
            if splitter in text:
                splitter_is_whitespace = False
                break

        # If no semantically meaningful splitter is present in the text, return an empty string as the splitter and the text as a list of characters.
        else:  # NOTE This code block will only be executed if the for loop completes without breaking.
            return "", splitter_is_whitespace, list(text)

    # Return the splitter and the split text.
    return splitter, splitter_is_whitespace, text.split(splitter)


def bisect_left(sorted: list, target: int, low: int, high: int) -> int:
    while low < high:
        mid = (low + high) // 2

        if sorted[mid] < target:
            low = mid + 1

        else:
            high = mid

    return low


def merge_splits(
    splits: list[str], cum_lens: list[int], chunk_size: int, splitter: str, token_counter: Callable, start: int, high: int
) -> tuple[int, str]:
    """Merge splits until a chunk size is reached, returning the index of the last split included in the merged chunk along with the merged chunk itself."""

    average = 0.2
    low = start

    offset = cum_lens[start]
    target = offset + (chunk_size * average)

    while low < high:
        i = bisect_left(cum_lens, target, low=low, high=high)
        midpoint = min(i, high - 1)

        tokens = token_counter(splitter.join(splits[start:midpoint]))

        local_cum = cum_lens[midpoint] - offset

        if local_cum and tokens > 0:
            average = local_cum / tokens
            target = offset + (chunk_size * average)

        if tokens > chunk_size:
            high = midpoint

        else:
            low = midpoint + 1

    end = low - 1
    return end, splitter.join(splits[start:end])


def chunk(
    text: str,
    chunk_size: int,
    token_counter: Callable[[str], int],
    memoize: bool = True,
    offsets: bool = False,
    overlap: float | int | None = None,
    cache_maxsize: int | None = None,
    _recursion_depth: int = 0,
    _start: int = 0,
) -> list[str] | tuple[list[str], list[tuple[int, int]]]:
    """Split a text into semantically meaningful chunks of a specified size as determined by the provided token counter.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum number of tokens a chunk may contain.
        token_counter (Callable[[str], int]): A callable that takes a string and returns the number of tokens in it.
        memoize (bool, optional): Whether to memoize the token counter. Defaults to `True`.
        offsets (bool, optional): Whether to return the start and end offsets of each chunk. Defaults to `False`.
        overlap (float | int | None, optional): The proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. Defaults to `None`, in which case no overlapping occurs.
        cache_maxsize (int | None, optional): The maximum number of text-token count pairs that can be stored in the token counter's cache. Defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

    Returns:
        list[str] | tuple[list[str], list[tuple[int, int]]]: A list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed, and, if `offsets` is `True`, a list of tuples of the form `(start, end)` where `start` is the index of the first character of the chunk in the original text and `end` is the index of the character after the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`."""

    # Rename variables for clarity.
    return_offsets = offsets
    local_chunk_size = chunk_size

    # If this is the first call, memoize the token counter if memoization is enabled and reduce the effective chunk size if overlapping chunks.
    if is_first_call := not _recursion_depth:
        if memoize:
            token_counter = _memoized_token_counters.setdefault(token_counter, lru_cache(cache_maxsize)(token_counter))

        if overlap:
            # Make relative overlaps absolute and floor both relative and absolute overlaps to prevent ever having an overlap >= chunk_size.
            overlap = math.floor(chunk_size * overlap) if overlap < 1 else min(overlap, chunk_size - 1)

            # If the overlap has not been zeroed, compute the effective chunk size as the minimum of the chunk size and the chunk size minus the overlap.
            if overlap:
                unoverlapped_chunk_size = chunk_size - overlap
                local_chunk_size = min(overlap, unoverlapped_chunk_size)

    # Split the text using the most semantically meaningful splitter possible.
    splitter, splitter_is_whitespace, splits = _split_text(text)

    offsets: list = []
    splitter_len = len(splitter)
    split_lens = [len(split) for split in splits]
    cum_lens = list(accumulate(split_lens, initial=0))
    split_starts = accumulate([0] + [split_len + splitter_len for split_len in split_lens])
    split_starts = [start + _start for start in split_starts]
    num_splits_plus_one = len(splits) + 1

    chunks = []
    skips = set()
    """A list of indices of splits to skip because they have already been added to a chunk."""

    # Iterate through the splits.
    for i, (split, split_start) in enumerate(zip(splits, split_starts)):
        # Skip the split if it has already been added to a chunk.
        if i in skips:
            continue

        # If the split is over the chunk size, recursively chunk it.
        if token_counter(split) > local_chunk_size:
            new_chunks, new_offsets = chunk(
                text=split,
                chunk_size=local_chunk_size,
                token_counter=token_counter,
                offsets=return_offsets,
                _recursion_depth=_recursion_depth + 1,
                _start=split_start,
            )

            chunks.extend(new_chunks)
            offsets.extend(new_offsets)

        # If the split is equal to or under the chunk size, add it and any subsequent splits to a new chunk until the chunk size is reached.
        else:
            # Merge the split with subsequent splits until the chunk size is reached.
            final_split_in_chunk_i, new_chunk = merge_splits(
                splits=splits,
                cum_lens=cum_lens,
                chunk_size=local_chunk_size,
                splitter=splitter,
                token_counter=token_counter,
                start=i,
                high=num_splits_plus_one,
            )

            # Mark any splits included in the new chunk for exclusion from future chunks.
            skips.update(range(i + 1, final_split_in_chunk_i))

            # Add the chunk.
            chunks.append(new_chunk)

            # Add the chunk's offsets.
            split_end = split_starts[final_split_in_chunk_i] - splitter_len
            offsets.append((split_start, split_end))

        # If the splitter is not whitespace and the split is not the last split, add the splitter to the end of the latest chunk if doing so would not cause it to exceed the chunk size otherwise add the splitter as a new chunk.
        if not splitter_is_whitespace and not (
            i == len(splits) - 1 or all(j in skips for j in range(i + 1, len(splits)))
        ):
            if token_counter(last_chunk_with_splitter := chunks[-1] + splitter) <= local_chunk_size:
                chunks[-1] = last_chunk_with_splitter
                start, end = offsets[-1]
                offsets[-1] = (start, end + splitter_len)

            else:
                start = offsets[-1][1] if offsets else split_start

                chunks.append(splitter)
                offsets.append((start, start + splitter_len))

    # If this is the first call, remove any empty chunks as well as chunks comprised entirely of whitespace and then overlap the chunks if desired and finally return the chunks, optionally with their offsets.
    if is_first_call:
        # Remove empty chunks.
        chunks_and_offsets = [(chunk, offset) for chunk, offset in zip(chunks, offsets) if chunk and not chunk.isspace()]
        
        if chunks_and_offsets:
            chunks, offsets = zip(*chunks_and_offsets)
            chunks, offsets = list(chunks), list(offsets)
        
        else:
            chunks, offsets = [], []

        # Overlap chunks if desired and there are chunks to overlap.
        if overlap and chunks:
            # Rename variables for clarity.
            subchunk_size = local_chunk_size
            subchunks = chunks
            suboffsets = offsets
            num_subchunks = len(subchunks)

            # Merge the subchunks into overlapping chunks.
            subchunks_per_chunk = math.floor(
                chunk_size / subchunk_size
            )  # NOTE `math.ceil` would cause the chunk size to be exceeded.
            subchunk_stride = math.floor(
                unoverlapped_chunk_size / subchunk_size
            )  # NOTE `math.ceil` would cause overlaps to be missed.

            offsets = [
                (
                    suboffsets[(start := i * subchunk_stride)][0],
                    suboffsets[min(start + subchunks_per_chunk, num_subchunks) - 1][1],
                )
                for i in range(max(1, math.ceil((num_subchunks - subchunks_per_chunk) / subchunk_stride) + 1))
            ]

            chunks = [text[start:end] for start, end in offsets]

        # Return offsets if desired.
        if return_offsets:
            return chunks, offsets

        return chunks

    # Always return chunks and offsets if this is a recursive call.
    return chunks, offsets


class Chunker:
    def __init__(self, chunk_size: int, token_counter: Callable[[str], int]) -> None:
        self.chunk_size = chunk_size
        self.token_counter = token_counter

    def _make_chunk_function(
        self,
        offsets: bool,
        overlap: float | int | None,
    ) -> Callable[[str], list[str] | tuple[list[str], list[tuple[int, int]]]]:
        """Construct a function that chunks a text and returns the chunks along with their offsets if necessary."""

        def _chunk(text: str) -> list[str] | tuple[list[str], list[tuple[int, int]]]:
            return chunk(
                text=text,
                chunk_size=self.chunk_size,
                token_counter=self.token_counter,
                memoize=False,
                offsets=offsets,
                overlap=overlap,
            )

        return _chunk

    def __call__(
        self,
        text_or_texts: str | Sequence[str],
        processes: int = 1,
        progress: bool = False,
        offsets: bool = False,
        overlap: int | float | None = None,
    ) -> (
        list[str]
        | tuple[list[str], list[tuple[int, int]]]
        | list[list[str]]
        | tuple[list[list[str]], list[list[tuple[int, int]]]]
    ):
        """Split text or texts into semantically meaningful chunks of a specified size as determined by the provided tokenizer or token counter.

        Args:
            text_or_texts (str | Sequence[str]): The text or texts to be chunked.
            processes (int, optional): The number of processes to use when chunking multiple texts. Defaults to `1` in which case chunking will occur in the main process.
            progress (bool, optional): Whether to display a progress bar when chunking multiple texts. Defaults to `False`.
            offsets (bool, optional): Whether to return the start and end offsets of each chunk. Defaults to `False`.
            overlap (float | int | None, optional): The proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. Defaults to `None`, in which case no overlapping occurs.

        Returns:
            list[str] | tuple[list[str], list[tuple[int, int]]] | list[list[str]] | tuple[list[list[str]], list[list[tuple[int, int]]]]: If a single text has been provided, a list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed, and, if `offsets` is `True`, a list of tuples of the form `(start, end)` where `start` is the index of the first character of the chunk in the original text and `end` is the index of the character succeeding the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

            If multiple texts have been provided, a list of lists of chunks, with each inner list corresponding to the chunks of one of the provided input texts, and, if `offsets` is `True`, a list of lists of tuples of the chunks' offsets to the original texts, as described above."""

        chunk_function = self._make_chunk_function(offsets=offsets, overlap=overlap)

        if isinstance(text_or_texts, str):
            return chunk_function(text_or_texts)

        if progress and processes == 1:
            text_or_texts = tqdm(text_or_texts)

        if processes == 1:
            chunks_and_offsets = [chunk_function(text) for text in text_or_texts]

        else:
            with mpire.WorkerPool(processes, use_dill=True) as pool:
                chunks_and_offsets = pool.map(chunk_function, text_or_texts, progress_bar=progress)

        if offsets:
            chunks, offsets_ = zip(*chunks_and_offsets)

            return list(chunks), list(offsets_)

        return chunks_and_offsets


def chunkerify(
    tokenizer_or_token_counter: str
    | tiktoken.Encoding
    | transformers.PreTrainedTokenizer
    | tokenizers.Tokenizer
    | Callable[[str], int],
    chunk_size: int | None = None,
    max_token_chars: int | None = None,
    memoize: bool = True,
    cache_maxsize: int | None = None,
) -> Chunker:
    """Construct a chunker that splits one or more texts into semantically meaningful chunks of a specified size as determined by the provided tokenizer or token counter.

    Args:
        tokenizer_or_token_counter (str | tiktoken.Encoding | transformers.PreTrainedTokenizer | tokenizers.Tokenizer | Callable[[str], int]): Either: the name of a `tiktoken` or `transformers` tokenizer (with priority given to the former); a tokenizer that possesses an `encode` attribute (e.g., a `tiktoken`, `transformers` or `tokenizers` tokenizer); or a token counter that returns the number of tokens in a input.
        chunk_size (int, optional): The maximum number of tokens a chunk may contain. Defaults to `None` in which case it will be set to the same value as the tokenizer's `model_max_length` attribute (deducted by the number of tokens returned by attempting to tokenize an empty string) if possible otherwise a `ValueError` will be raised.
        max_token_chars (int, optional): The maximum number of characters a token may contain. Used to significantly speed up the token counting of long inputs. Defaults to `None` in which case it will either not be used or will, if possible, be set to the number of characters in the longest token in the tokenizer's vocabulary as determined by the `token_byte_values` or `get_vocab` methods.
        memoize (bool, optional): Whether to memoize the token counter. Defaults to `True`.
        cache_maxsize (int, optional): The maximum number of text-token count pairs that can be stored in the token counter's cache. Defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

    Returns:
        Callable[[str | Sequence[str], bool, bool, bool, int | float | None], list[str] | tuple[list[str], list[tuple[int, int]]] | list[list[str]] | tuple[list[list[str]], list[list[tuple[int, int]]]]]: A chunker that takes either a single text or a sequence of texts and returns, depending on whether multiple texts have been provided, a list or list of lists of chunks up to `chunk_size`-tokens-long with any whitespace used to split the text removed, and, if the optional `offsets` argument to the chunker is `True`, a list or lists of tuples of the form `(start, end)` where `start` is the index of the first character of a chunk in a text and `end` is the index of the character succeeding the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

        The resulting chunker can be passed a `processes` argument that specifies the number of processes to be used when chunking multiple texts.

        It is also possible to pass a `progress` argument which, if set to `True` and multiple texts are passed, will display a progress bar.

        As described above, the `offsets` argument, if set to `True`, will cause the chunker to return the start and end offsets of each chunk.

        The chunker accepts an `overlap` argument that specifies the proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. It defaults to `None`, in which case no overlapping occurs."""

    # If the provided tokenizer is a string, try to load it with either `tiktoken` or `transformers` or raise an error if neither is available.
    if isinstance(tokenizer_or_token_counter, str):
        try:
            import tiktoken

            try:
                tokenizer = tiktoken.encoding_for_model(tokenizer_or_token_counter)

            except Exception:
                tokenizer = tiktoken.get_encoding(tokenizer_or_token_counter)

        except Exception:
            try:
                import transformers

                tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_or_token_counter)

            except Exception:
                raise ValueError(
                    f'"{tokenizer_or_token_counter}" was provided to `semchunk.chunkerify` as the name of a tokenizer but neither `tiktoken` nor `transformers` have a tokenizer by that name. Perhaps they are not installed or maybe there is a typo in that name?'
                )

        tokenizer_or_token_counter = tokenizer

    # If the number of characters in the longest token has not been provided, determine it if possible.
    if max_token_chars is None:
        for potential_vocabulary_getter_function in (
            "token_byte_values",  # Employed by `tiktoken`.
            "get_vocab",  # Employed by `tokenizers`.
        ):
            if hasattr(tokenizer_or_token_counter, potential_vocabulary_getter_function) and callable(
                getattr(tokenizer_or_token_counter, potential_vocabulary_getter_function)
            ):
                vocab = getattr(tokenizer_or_token_counter, potential_vocabulary_getter_function)()

                if hasattr(vocab, "__iter__") and vocab and all(hasattr(token, "__len__") for token in vocab):
                    max_token_chars = max(len(token) for token in vocab)
                    break

    # If a chunk size has not been specified, set it to the maximum number of tokens the tokenizer supports if possible otherwise raise an error.
    if chunk_size is None:
        if hasattr(tokenizer_or_token_counter, "model_max_length") and isinstance(
            tokenizer_or_token_counter.model_max_length, int
        ):
            chunk_size = tokenizer_or_token_counter.model_max_length

            # Attempt to reduce the chunk size by the number of special characters typically added by the tokenizer.
            if hasattr(tokenizer_or_token_counter, "encode"):
                with suppress(Exception):
                    chunk_size -= len(tokenizer_or_token_counter.encode(""))

        else:
            raise ValueError(
                "Your desired chunk size was not passed to `semchunk.chunkerify` and the provided tokenizer either lacks an attribute named 'model_max_length' or that attribute is not an integer. Either specify a chunk size or provide a tokenizer that has a 'model_max_length' attribute that is an integer."
            )

    # If we have been given a tokenizer, construct a token counter from it.
    if hasattr(tokenizer_or_token_counter, "encode"):
        # Determine whether the tokenizer accepts the argument `add_special_tokens` and, if so, ensure that it is always disabled.
        if "add_special_tokens" in inspect.signature(tokenizer_or_token_counter.encode).parameters:

            def token_counter(text: str) -> int:
                return len(tokenizer_or_token_counter.encode(text, add_special_tokens=False))

        else:

            def token_counter(text: str) -> int:
                return len(tokenizer_or_token_counter.encode(text))

    else:
        token_counter = tokenizer_or_token_counter

    # If we know the number of characters in the longest token, construct a new token counter that uses that to avoid having to tokenize very long texts.
    if max_token_chars is not None:
        max_token_chars = max_token_chars - 1
        original_token_counter = token_counter

        def faster_token_counter(text: str) -> int:
            heuristic = chunk_size * 6

            if len(text) > heuristic and original_token_counter(text[: heuristic + max_token_chars]) > chunk_size:
                return chunk_size + 1

            return original_token_counter(text)

        token_counter = faster_token_counter

    # Memoize the token counter if necessary.
    if memoize:
        token_counter = _memoized_token_counters.setdefault(token_counter, lru_cache(cache_maxsize)(token_counter))

    # Construct and return the chunker.
    return Chunker(chunk_size=chunk_size, token_counter=token_counter)

from __future__ import annotations

import os
import re
import math
import inspect

from bisect import bisect_left
from typing import Callable, Sequence, TypeAlias, TYPE_CHECKING
from functools import partial, lru_cache
from itertools import accumulate
from contextlib import suppress

import mpire

from tqdm import tqdm

try:
    import isaacus as isaacus_runtime

    from isaacus.types.ilgs.v1.document import Document as ILGSDocument_Runtime

except ImportError:
    isaacus_runtime = None
    ILGSDocument_Runtime = None

if TYPE_CHECKING:
    import isaacus
    import tiktoken
    import tokenizers
    import transformers

    from isaacus.types.ilgs.v1.document import Document as ILGSDocument


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

_REGEX_ESCAPED_NON_WHITESPACE_SEMANTIC_SPLITTERS = tuple(
    re.escape(splitter) for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS
)


def _split_text(text: str) -> tuple[str, bool, list[str]]:
    """Split text using the most semantically meaningful splitter possible."""

    splitter_is_whitespace = True

    # Try splitting at, in order of most desirable to least desirable:
    # - The largest sequence of newlines and/or carriage returns;
    # - The largest sequence of tabs;
    # - The largest sequence of whitespace characters or, if the largest such sequence is only a single character and there exists a whitespace character preceded by a semantically meaningful non-whitespace splitter, then that whitespace character;
    # - A semantically meaningful non-whitespace splitter.
    if "\n" in text or "\r" in text:
        splitter = max(re.findall(r"[\r\n]+", text), key=len)

    elif "\t" in text:
        splitter = max(re.findall(r"\t+", text), key=len)

    elif re.search(r"\s", text):
        splitter = max(re.findall(r"\s+", text), key=len)

        # If the splitter is only a single character, see if we can target whitespace characters that are preceded by semantically meaningful non-whitespace splitters to avoid splitting in the middle of sentences.
        if len(splitter) == 1:
            for escaped_preceder in _REGEX_ESCAPED_NON_WHITESPACE_SEMANTIC_SPLITTERS:
                if whitespace_preceded_by_preceder := re.search(rf"{escaped_preceder}(\s)", text):
                    splitter = whitespace_preceded_by_preceder.group(1)
                    escaped_splitter = re.escape(splitter)

                    return (
                        splitter,
                        splitter_is_whitespace,
                        re.split(rf"(?<={escaped_preceder}){escaped_splitter}", text),
                    )

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


def merge_splits(
    text: str,
    split_starts: Sequence[int],
    splitter_len: int,
    cum_lens: Sequence[int],
    chunk_size: int,
    token_counter: Callable[[str], int],
    start: int,
    high: int,
) -> tuple[int, str]:
    """Merge splits until a chunk size is reached, returning the index of the first split not included in the merged chunk along with the merged chunk itself."""

    average = 0.2
    low = start

    offset = cum_lens[start]
    target = offset + (chunk_size * average)

    while low < high:
        i = bisect_left(cum_lens, target, lo=low, hi=high)
        midpoint = min(i, high - 1)

        tokens = token_counter(text[split_starts[start] : split_starts[midpoint] - splitter_len])

        local_cum = cum_lens[midpoint] - offset

        if local_cum and tokens > 0:
            average = local_cum / tokens
            target = offset + (chunk_size * average)

        if tokens > chunk_size:
            high = midpoint

        else:
            low = midpoint + 1

    end = low - 1
    return end, (text[split_starts[start] : split_starts[end] - splitter_len])


SpanNode: TypeAlias = tuple[tuple[int, int], list["SpanNode"], "SpanNode | None"]


def chunk(
    text: "str | ILGSDocument",
    chunk_size: int,
    token_counter: Callable[[str], int],
    *,
    memoize: bool = True,
    offsets: bool = False,
    overlap: float | int | None = None,
    chunking_model: str | None = None,
    isaacus_client: "isaacus.Isaacus | None" = None,
    cache_maxsize: int | None = None,
    _recursion_depth: int = 0,
    _start: int = 0,
) -> list[str] | tuple[list[str], list[tuple[int, int]]]:
    """Split a text into semantically meaningful chunks of a specified size as determined by the provided token counter.

    Args:
        text (str | ILGSDocument): The input to be chunked. If you pass an Isaacus Legal Graph Schema (ILGS) Document, AI chunking will occur automatically without re-enriching the document.

        chunk_size (int): The maximum number of tokens a chunk may contain.

        token_counter (Callable[[str], int]): A callable that takes a string and returns the number of tokens in it.

        memoize (bool, optional): Whether to memoize the token counter. Defaults to `True`.

        offsets (bool, optional): Whether to return the start and end offsets of each chunk. Defaults to `False`.

        overlap (float | int | None, optional): The proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. Defaults to `None`, in which case no overlapping occurs.

        chunking_model (str, optional): The name of the Isaacus enrichment model to use for AI chunking. Defaults to `None`, in which case, unless you provide an Isaacus Legal Graph Schema (ILGS) Document as input, AI chunking will be disabled.

        isaacus_client (isaacus.Isaacus, optional): An instance of the `isaacus.Isaacus` API client to use for AI chunking instead of a client constructed with default parameters. Defaults to `None`, in which case a client will be constructed with default parameters if AI chunking is enabled.

        cache_maxsize (int | None, optional): The maximum number of text-token count pairs that can be stored in the token counter's cache. Defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

    Returns:
        list[str] | tuple[list[str], list[tuple[int, int]]]: A list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed, and, if `offsets` is `True`, a list of tuples of the form `(start, end)` where `start` is the index of the first character of the chunk in the original text and `end` is the index of the character after the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`."""

    # region ### Initialization ###
    # Rename variables for clarity.
    return_offsets = offsets
    local_chunk_size = chunk_size
    ilgs_doc = None

    if is_first_call := not _recursion_depth:
        # Memoize the token counter if desired.
        if memoize:
            token_counter = _memoized_token_counters.setdefault(token_counter, lru_cache(cache_maxsize)(token_counter))

        # Reduce the effective chunk size if overlapping is enabled.
        if overlap:
            # Make relative overlaps absolute and floor both relative and absolute overlaps to prevent ever having an overlap over the chunk size.
            overlap = math.floor(chunk_size * overlap) if overlap < 1 else min(overlap, chunk_size - 1)

            # If the overlap has not been zeroed, compute the effective chunk size as the minimum of the overlap and the chunk size minus the overlap.
            if overlap:
                unoverlapped_chunk_size = chunk_size - overlap
                local_chunk_size = min(overlap, unoverlapped_chunk_size)

        # If the input is an ILGS Document, stash it and set its text to the text to be chunked.
        if isaacus_runtime is not None and isinstance(text, ILGSDocument_Runtime):
            ilgs_doc = text
            text = ilgs_doc.text

    # endregion

    # region ### AI chunking ###
    # If the input is an ILGS Document or if a `chunking_model` was provided, and the text exceeds the chunk size and is not composed entirely of whitespace, perform AI chunking.
    if (
        (ilgs_doc is not None or chunking_model is not None)
        and token_counter(text) > local_chunk_size
        and not text.isspace()
    ):
        # If an ILGS Document was not supplied, leverage the chunking model to enrich the input.
        if ilgs_doc is None:
            # If an `isaacus_client` was not supplied, create one.
            if isaacus_client is None:
                try:
                    import isaacus

                except ImportError as e:
                    raise ImportError(
                        """AI chunking requires the Isaacus SDK to be installed. Please install it with `pip install isaacus`."""
                    ) from e

                api_key = os.getenv("ISAACUS_API_KEY")

                if api_key is None:
                    raise ValueError(
                        """AI chunking requires the `ISAACUS_API_KEY` environment variable to be set. Obtain an API key at https://platform.isaacus.com/accounts/signup/. Then run `import os; os.environ["ISAACUS_API_KEY"] = "your_api_key"` before calling Semchunk."""
                    )

                isaacus_client = isaacus.Isaacus()

            # If the input text has over 1,000,000 characters, to avoid timeouts and rate limit errors, we can split it up into 1,000,000-character-long chunks and join the results back together.
            if len(text) > 1_000_000:
                prechunks, prechunk_offsets = chunk(
                    text, chunk_size=1_000_000, token_counter=len, memoize=False, offsets=True
                )

            else:
                prechunks = [text]
                prechunk_offsets = [(0, len(text))]

            # Hierarchically segment the prechunk with the chunking model.
            enriched_prechunks = [
                isaacus_client.enrichments.create(model=chunking_model, texts=[prechunk], overflow_strategy="auto")
                .results[0]
                .document
                for prechunk in prechunks
            ]
        
        else:
            enriched_prechunks = [ilgs_doc]
            prechunk_offsets = [(0, len(text))]
        
        # Hierarchically segment each prechunk, constructing a tree out of extracted spans.
        span_tree: SpanNode = ((0, len(text)), [], None)

        for enriched_prechunk, (prechunk_start, prechunk_end) in zip(enriched_prechunks, prechunk_offsets):
            # Extract all spans from the enriched prechunk, adjusting their indices to be relative to the original input text, sorting them by (start, -end) such that children come after their parents.
            # NOTE Spans are zero-based, half-open, Unicode code point-spaced indices. They cannot start or end at whitespace (though because ends are exclusive, the 'end' index can occur at whitespace), cannot be empty, and cannot partially overlap (they are well-nested and globally laminar).
            prechunk_spans = sorted(
                {
                    (span.start + prechunk_start, span.end + prechunk_start)
                    for span in (
                        *[
                            span
                            for seg in enriched_prechunk.segments
                            for span in (seg.type_name, seg.code, seg.title, seg.span)
                        ],
                        enriched_prechunk.title,
                        enriched_prechunk.subtitle,
                        *[xref.span for xref in enriched_prechunk.crossreferences],
                        *[
                            m
                            for mentionable in (
                                enriched_prechunk.locations,
                                enriched_prechunk.persons,
                                enriched_prechunk.emails,
                                enriched_prechunk.websites,
                                enriched_prechunk.phone_numbers,
                                enriched_prechunk.id_numbers,
                                enriched_prechunk.terms,
                                enriched_prechunk.external_documents,
                                enriched_prechunk.dates,
                            )
                            for ent in mentionable
                            for m in ent.mentions
                        ],
                        *[s for term in enriched_prechunk.terms for s in (term.name, term.meaning)],
                        *[p for exd in enriched_prechunk.external_documents for p in exd.pinpoints],
                        *[quote.span for quote in enriched_prechunk.quotes],
                        *enriched_prechunk.headings,
                        *enriched_prechunk.junk,
                    )
                    if span is not None
                },
                key=lambda span: (span[0], -span[1]),
            )

            # Build a tree out of the spans, creating new spans at each level of depth where necessary in order to cover content not covered by any existing spans at that depth.
            parent_node: SpanNode = span_tree
            last_end_by_node: dict[int, int] = {id(span_tree): prechunk_start}

            for span_start, span_end in prechunk_spans:
                # Traverse up the tree until we find the parent of the current span, creating new spans at each level of depth where necessary.
                while span_end > parent_node[0][1]:
                    pid = id(parent_node)
                    node_last_end = last_end_by_node.get(pid)

                    if node_last_end is not None and node_last_end < parent_node[0][1]:
                        parent_node[1].append(((node_last_end, parent_node[0][1]), [], parent_node))

                    parent_node = parent_node[2]

                # If the span starts after the end of the last span, add a new span covering the content in between.
                parent_id = id(parent_node)
                parent_last_end = last_end_by_node.get(parent_id, parent_node[0][0])

                if span_start > parent_last_end:
                    parent_node[1].append(((parent_last_end, span_start), [], parent_node))

                # Add the current span as a child of its parent.
                curr_node: SpanNode = ((span_start, span_end), [], parent_node)
                parent_node[1].append(curr_node)

                last_end_by_node[parent_id] = span_end
                parent_node = curr_node

            # Add new spans covering any remaining uncovered content.
            while parent_node is not None and parent_node is not span_tree:
                pid = id(parent_node)
                node_last_end = last_end_by_node.get(pid)

                if node_last_end is not None and node_last_end < parent_node[0][1]:
                    parent_node[1].append(((node_last_end, parent_node[0][1]), [], parent_node))

                parent_node = parent_node[2]

            root_last_end = last_end_by_node.get(id(span_tree), prechunk_start)

            if root_last_end < prechunk_end:
                span_tree[1].append(((root_last_end, prechunk_end), [], span_tree))

        # Chunk the text.
        offsets: list[tuple[int, int]] = []
        chunk_start = 0
        chunk_end = None
        stack: list[SpanNode] = [span_tree]

        while stack:
            (node_start, node_end), node_children, _ = stack.pop()

            # If the current node can be added to the current chunk without exceeding the chunk size, add it.
            new_chunk_text = text[chunk_start:node_end]

            if token_counter(new_chunk_text) <= local_chunk_size:
                chunk_end = node_end

            # Otherwise, if the current node cannot fit within the current chunk but the resulting chunk would be composed entirely of whitespace anyways, discard the current chunk and node and start a new chunk after this node.
            elif new_chunk_text.isspace():
                chunk_start = node_end
                chunk_end = None

            # Otherwise, if the current node cannot fit within the current chunk, add the current chunk as a completed chunk if it is not composed entirely of whitespace, and then, if the current node does not exceed the chunk size on its own, start a new chunk with the current node, otherwise, if the current node has children, add them to the stack to be processed, otherwise, naively chunk the current node.
            else:
                if chunk_end is not None and not text[chunk_start:chunk_end].isspace():
                    # Remove trailing whitespace from the chunk.
                    while text[chunk_end - 1].isspace():
                        chunk_end -= 1

                    # Remove leading whitespace from the chunk.
                    while text[chunk_start].isspace():
                        chunk_start += 1

                    offsets.append((chunk_start, chunk_end))

                if token_counter(text[node_start:node_end]) <= local_chunk_size:
                    chunk_start = node_start
                    chunk_end = node_end

                elif node_children:
                    chunk_start = node_start
                    chunk_end = None
                    stack.extend(reversed(node_children))

                else:
                    _, naive_offsets = chunk(
                        text[node_start:node_end],
                        chunk_size=local_chunk_size,
                        token_counter=token_counter,
                        memoize=False,
                        offsets=True,
                    )

                    if naive_offsets:
                        last_offset_i = len(naive_offsets) - 1

                        for i, (naive_start, naive_end) in enumerate(naive_offsets):
                            naive_start = node_start + naive_start
                            naive_end = node_start + naive_end

                            if i == last_offset_i:
                                chunk_start = naive_start
                                chunk_end = node_end

                            else:
                                # Remove trailing whitespace from the chunk.
                                while text[naive_end - 1].isspace():
                                    naive_end -= 1

                                # Remove leading whitespace from the chunk.
                                while text[naive_start].isspace():
                                    naive_start += 1

                                offsets.append((naive_start, naive_end))

        # Add the final chunk if it is not composed entirely of whitespace.
        if chunk_end is not None and not text[chunk_start:chunk_end].isspace():
            # Remove trailing whitespace from the chunk.
            while text[chunk_end - 1].isspace():
                chunk_end -= 1

            # Remove leading whitespace from the chunk.
            while text[chunk_start].isspace():
                chunk_start += 1

            offsets.append((chunk_start, chunk_end))

        # Overlap chunks if desired.
        if overlap and offsets:
            # Rename variables for clarity.
            subchunk_size = local_chunk_size
            suboffsets = offsets
            num_subchunks = len(suboffsets)

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

        # Materialize chunks from offsets.
        chunks = [text[start:end] for start, end in offsets]

        # Return chunks, and, if desired, offsets.
        if return_offsets:
            return chunks, offsets

        return chunks

    # endregion

    # region ### Chunking ###
    # Split the text using the most semantically meaningful splitter possible.
    splitter, splitter_is_whitespace, splits = _split_text(text)

    offsets: list = []
    splitter_len = len(splitter)
    split_lens = [len(split) for split in splits]
    local_split_starts = list(accumulate([0] + [split_len + splitter_len for split_len in split_lens]))
    split_starts = [start + _start for start in local_split_starts]
    num_splits_plus_one = len(splits) + 1

    chunks = []
    skips = set()
    """A set of indices of splits to skip because they have already been added to a chunk."""

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
                offsets=True,
                _recursion_depth=_recursion_depth + 1,
                _start=split_start,
            )

            chunks.extend(new_chunks)
            offsets.extend(new_offsets)

        # If the split is equal to or under the chunk size, add it and any subsequent splits to a new chunk until the chunk size is reached.
        else:
            # Merge the split with subsequent splits until the chunk size is reached.
            final_split_in_chunk_i, new_chunk = merge_splits(
                text=text,
                split_starts=local_split_starts,
                splitter_len=splitter_len,
                cum_lens=local_split_starts,
                chunk_size=local_chunk_size,
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
        chunks_and_offsets = [
            (chunk, offset) for chunk, offset in zip(chunks, offsets) if chunk and not chunk.isspace()
        ]

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
    # endregion


class Chunker:
    def __init__(
        self,
        chunk_size: int,
        token_counter: Callable[[str], int],
        chunking_model: str | None = None,
        isaacus_client: "isaacus.Isaacus | None" = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.token_counter = token_counter
        self.chunking_model = chunking_model
        self.isaacus_client = isaacus_client

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
                chunking_model=self.chunking_model,
                isaacus_client=self.isaacus_client,
                memoize=False,
                offsets=offsets,
                overlap=overlap,
            )

        return _chunk

    def __call__(
        self,
        text_or_texts: "str | ILGSDocument | Sequence[str | ILGSDocument]",
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
            text_or_texts (str | ILGSDocument | Sequence[str | ILGSDocument]): The input or inputs to be chunked. For inputs that are Isaacus Legal Graph Schema (ILGS) Documents, AI chunking will occur automatically without re-enriching the documents.

            processes (int, optional): The number of processes to use when chunking multiple texts. Defaults to `1` in which case chunking will occur in the main process.

            progress (bool, optional): Whether to display a progress bar when chunking multiple texts. Defaults to `False`.

            offsets (bool, optional): Whether to return the start and end offsets of each chunk. Defaults to `False`.

            overlap (float | int | None, optional): The proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. Defaults to `None`, in which case no overlapping occurs.

        Returns:
            list[str] | tuple[list[str], list[tuple[int, int]]] | list[list[str]] | tuple[list[list[str]], list[list[tuple[int, int]]]]: If a single text has been provided, a list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed, and, if `offsets` is `True`, a list of tuples of the form `(start, end)` where `start` is the index of the first character of the chunk in the original text and `end` is the index of the character succeeding the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

            If multiple texts have been provided, a list of lists of chunks, with each inner list corresponding to the chunks of one of the provided input texts, and, if `offsets` is `True`, a list of lists of tuples of the chunks' offsets to the original texts, as described above."""

        chunk_function = self._make_chunk_function(offsets=offsets, overlap=overlap)

        if isinstance(text_or_texts, str) or (isaacus_runtime is not None and isinstance(text_or_texts, ILGSDocument_Runtime)):
            return chunk_function(text_or_texts)

        if progress and processes == 1:
            text_or_texts = tqdm(text_or_texts)

        if processes == 1:
            chunks_and_offsets = [chunk_function(text) for text in text_or_texts]

        else:
            with mpire.WorkerPool(processes, use_dill=True) as pool:
                chunks_and_offsets = pool.map(chunk_function, [(text,) for text in text_or_texts], progress_bar=progress)

        if offsets:
            chunks, offsets_ = zip(*chunks_and_offsets)

            return list(chunks), list(offsets_)

        return chunks_and_offsets


def chunkerify(
    tokenizer_or_token_counter: "str | tiktoken.Encoding | transformers.PreTrainedTokenizer | tokenizers.Tokenizer | Callable[[str], int]",
    chunk_size: int | None = None,
    *,
    chunking_model: str | None = None,
    isaacus_client: "isaacus.Isaacus | None" = None,
    tokenizer_kwargs: dict | None = None,
    max_token_chars: int | None = None,
    memoize: bool = True,
    cache_maxsize: int | None = None,
) -> Chunker:
    """Construct a chunker that splits one or more texts into semantically meaningful chunks of a specified size as determined by the provided tokenizer or token counter.

    Args:
        tokenizer_or_token_counter (str | tiktoken.Encoding | transformers.PreTrainedTokenizer | tokenizers.Tokenizer | Callable[[str], int]): Either: the name of a Tiktoken or Transformers tokenizer (with priority given to the former); a tokenizer with an `encode()` method (e.g., a `tiktoken` or Transformers tokenizer); or a token counter that returns the number of tokens in an input.

        chunk_size (int, optional): The maximum number of tokens a chunk may contain. Defaults to `None`, in which case it will be set to the same value as the tokenizer's `model_max_length` attribute (minus the number of tokens returned by attempting to tokenize an empty string) if possible otherwise a `ValueError` will be raised.

        chunking_model (str, optional): The name of the Isaacus enrichment model to use for AI chunking. Defaults to `None`, in which case, unless you provide an Isaacus Legal Graph Schema (ILGS) Document as input, AI chunking will be disabled.

        isaacus_client (isaacus.Isaacus, optional): An instance of the `isaacus.Isaacus` API client to use for AI chunking instead of a client constructed with default parameters. Defaults to `None`, in which case a client will be constructed with default parameters if AI chunking is enabled.

        tokenizer_kwargs (dict, optional): A dictionary of keyword arguments to be passed to the tokenizer or token counter whenever it is called. This can be used to disable the current default behavior of treating any encountered special tokens as if they are normal text when using a Tiktoken or Transformers tokenizer. Defaults to `None`, in which case no additional keyword arguments will be passed to the tokenizer or token counter.

        max_token_chars (int, optional): The maximum number of characters a token may contain. Used to significantly speed up the token counting of long inputs. Defaults to `None` in which case it will either not be used or will, if possible, be set to the number of characters in the longest token in the tokenizer's vocabulary as determined by the `token_byte_values` or `get_vocab` methods.

        memoize (bool, optional): Whether to memoize the token counter. Defaults to `True`.

        cache_maxsize (int, optional): The maximum number of text-token count pairs that can be stored in the token counter's cache. Defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

    Returns:
        Callable[[str | Sequence[str], bool, bool, bool, int | float | None], list[str] | tuple[list[str], list[tuple[int, int]]] | list[list[str]] | tuple[list[list[str]], list[list[tuple[int, int]]]]]: A chunker that takes either a single text or a sequence of texts and returns, depending on whether multiple texts have been provided, a list or list of lists of chunks up to `chunk_size`-tokens-long with any whitespace used to split the text removed, and, if the optional `offsets` argument to the chunker is `True`, a list or lists of tuples of the form `(start, end)` where `start` is the index of the first character of a chunk in a text and `end` is the index of the character succeeding the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

        The resulting chunker can be passed a `processes` argument that specifies the number of processes to be used when chunking multiple texts.

        It is also possible to pass a `progress` argument which, if set to `True` and multiple texts are passed, will display a progress bar.

        As described above, the `offsets` argument, if set to `True`, will cause the chunker to return the start and end offsets of each chunk.

        The chunker accepts an `overlap` argument that specifies the proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. It defaults to `None`, in which case no overlapping occurs."""

    # If the provided tokenizer is a string, try to load it with either Tiktoken or Transformers or raise an error if neither is available.
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
                    f'"{tokenizer_or_token_counter}" was provided to `semchunk.chunkerify` as the name of a tokenizer but neither Tiktoken nor Transformers have a tokenizer by that name. Perhaps they are not installed or maybe there is a typo in that name?'
                )

        tokenizer_or_token_counter = tokenizer

    # If the number of characters in the longest token has not been provided, determine it if possible.
    if max_token_chars is None:
        for potential_vocabulary_getter_function in (
            "token_byte_values",  # Employed by Tiktoken.
            "get_vocab",  # Employed by Transformers and Tokenizers.
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

    # If we have been given a tokenizer, construct a token counter from it, otherwise, assume we have been given a token counter directly.
    if hasattr(tokenizer_or_token_counter, "encode"):
        # If the tokenizer accepts keyword arguments allowing us to specify that special tokens should not be added and should be treated as normal text and those arguments are not already being disabled by `tokenizer_kwargs`, set them.
        tokenizer_kwargs = dict(tokenizer_kwargs) if tokenizer_kwargs is not None else {}

        try:
            tokenizer_parameters = inspect.signature(tokenizer_or_token_counter.encode).parameters

        except Exception:
            tokenizer_parameters = {}

        for kwarg, value in {
            "add_special_tokens": False,
            "split_special_tokens": True,
            "disallowed_special": (),
        }.items():
            if kwarg in tokenizer_parameters and kwarg not in tokenizer_kwargs:
                tokenizer_kwargs[kwarg] = value

        if tokenizer_kwargs:

            def token_counter(text: str) -> int:
                return len(tokenizer_or_token_counter.encode(text, **tokenizer_kwargs))

        else:

            def token_counter(text: str) -> int:
                return len(tokenizer_or_token_counter.encode(text))

    else:
        if tokenizer_kwargs:
            token_counter = partial(tokenizer_or_token_counter, **tokenizer_kwargs)

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
    return Chunker(
        chunk_size=chunk_size, token_counter=token_counter, chunking_model=chunking_model, isaacus_client=isaacus_client
    )

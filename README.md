<div align='center'>

# semchunk 🧩

<a href="https://pypi.org/project/semchunk/"><img src="https://img.shields.io/pypi/v/semchunk" alt="PyPI version" /></a> <a href="https://github.com/isaacus-dev/semchunk/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/isaacus-dev/semchunk/ci.yml?branch=main" alt="Build status"></a> <a href="https://app.codecov.io/gh/isaacus-dev/semchunk"><img src="https://img.shields.io/codecov/c/github/isaacus-dev/semchunk" alt="Code coverage"></a> <a href="https://pypistats.org/packages/semchunk"><img src="https://img.shields.io/pypi/dm/semchunk" alt="Monthly downloads"></a>

</div>

**semchunk** is a Python library for splitting text into smaller chunks while preserving as much local semantic context as possible.

semchunk supports AI-powered chunking, chunk overlapping, and chunk offsets, and works with any tokenizer or token counter, including those from Tiktoken and Transformers.

Powered by a novel hierarchical chunking algorithm, semchunk is both more accurate and faster than naïve approaches (see [How It Works 🔍](https://github.com/isaacus-dev/semchunk#how-it-works-)).

semchunk is production-ready. It is downloaded millions of times per month and is used in Docling, the Microsoft Intelligence Toolkit, and the Isaacus API.

## Setup 📦

semchunk can be installed with `pip` (or `uv`):

```bash
pip install semchunk
```

If you're using AI-powered chunking, you'll also want to install the [Isaacus SDK](https://github.com/isaacus-dev/isaacus-python) and obtain an [Isaacus API key](https://platform.isaacus.com/accounts/signup/):

```bash
pip install isaacus
```

semchunk is also available on `conda-forge`:

```bash
conda install conda-forge::semchunk
# or
conda install -c conda-forge semchunk
```

[@dominictarro](https://github.com/dominictarro) maintains a Rust port of semchunk named [`semchunk-rs`](https://crates.io/crates/semchunk-rs).

## Quickstart 👩‍💻

The code snippet below demonstrates how to chunk text with semchunk:

```python
import semchunk
import tiktoken                        # Transformers and Tiktoken are not dependencies,
from transformers import AutoTokenizer # they're just here for demonstration purposes.

chunk_size = 4 # A low chunk size is used here for demonstration purposes. Keep in mind, semchunk
               # does not know how many special tokens, if any, your tokenizer adds to every input,
               # so you may want to deduct the number of special tokens added from your chunk size.
text = 'The quick brown fox jumps over the lazy dog.'

# You can construct a chunker with `semchunk.chunkerify()` by passing the name of an OpenAI model,
# OpenAI Tiktoken encoding or Hugging Face model, or a custom tokenizer that has an `encode()` method
# (like a Tiktoken or Transformers tokenizer) or a custom token counting function that takes a text and
# returns the number of tokens in it.
chunker = semchunk.chunkerify('isaacus/kanon-2-tokenizer', chunk_size) or \
          semchunk.chunkerify('gpt-4', chunk_size) or \
          semchunk.chunkerify('cl100k_base', chunk_size) or \
          semchunk.chunkerify(AutoTokenizer.from_pretrained('isaacus/kanon-2-tokenizer'), chunk_size) or \
          semchunk.chunkerify(tiktoken.encoding_for_model('gpt-4'), chunk_size) or \
          semchunk.chunkerify(lambda text: len(text.split()), chunk_size)

# If you give the resulting chunker a single text, it'll return a list of chunks. If you give it a
# list of texts, it'll return a list of lists of chunks.
assert chunker(text) == ['The quick brown fox', 'jumps over the', 'lazy dog.']
assert chunker([text], progress=True) == [['The quick brown fox', 'jumps over the', 'lazy dog.']]

# If you have a lot of texts and you want to speed things up, you can enable multiprocessing by
# setting `processes` to a number greater than 1.
assert chunker([text], processes=2) == [['The quick brown fox', 'jumps over the', 'lazy dog.']]

# You can also pass an `offsets` argument to return the offsets of chunks, as well as an `overlap`
# argument to overlap chunks by a ratio (if < 1) or an absolute number of tokens (if >= 1).
chunks, offsets = chunker(text, offsets=True, overlap=0.5)
```

To leverage AI-powered chunking, ensure the [Isaacus SDK](https://github.com/isaacus-dev/isaacus-python) is installed and your `ISAACUS_API_KEY` environment variable is set, and then simply pass the name of an [Isaacus enrichment model](https://docs.isaacus.com/models/introduction#enrichment) like `kanon-2-enricher` as the `chunking_model` argument of `chunkerify()` or `chunk()`, like so:

```python
import requests  # For demonstration purposes, we'll use `requests` to download a long document.
import semchunk
from os import environ

# Set your `ISAACUS_API_KEY` environment variable to your Isaacus API key.
environ["ISAACUS_API_KEY"] = "INSERT_YOUR_API_KEY_HERE"

# Download a very long document to chunk.
text = requests.get("https://examples.isaacus.com/dred-scott-v-sandford.txt").text

# Construct a chunker that uses `kanon-2-enricher` for AI-powered chunking.
# NOTE Because we're using a Hugging Face Transformers tokenizer, the `transformers` library is required here,
# however, you can use any tokenizer or token counter you like.
chunker = semchunk.chunkerify("isaacus/kanon-2-tokenizer", 512, chunking_model="kanon-2-enricher")

# Chunk the document with AI-powered chunking.
chunks = chunker(text)
```

## Usage 🕹️

### `chunkerify()`

```python
def chunkerify(
    tokenizer_or_token_counter: str | tiktoken.Encoding | transformers.PreTrainedTokenizer | \
                                tokenizers.Tokenizer | Callable[[str], int],
    chunk_size: int | None = None,
    *,
    chunking_model: str | None = None,
    isaacus_client: isaacus.Isaacus | None = None,
    tokenizer_kwargs: dict | None = None,
    max_token_chars: int | None = None,
    memoize: bool = True,
    cache_maxsize: int | None = None,
) -> Callable[[str | ILGSDocument | Sequence[str | ILGSDocument], int, bool, bool, int | float | None], list[str] | tuple[list[str], list[tuple[int, int]]] | list[list[str]] | tuple[list[list[str]], list[list[tuple[int, int]]]]]:
```

`chunkerify()` constructs a chunker that splits one or more texts into semantically meaningful chunks of a specified size as determined by the provided tokenizer or token counter.

`tokenizer_or_token_counter` is either: the name of a Tiktoken or Transformers tokenizer (with priority given to the former); a tokenizer with an `encode()` method (e.g., a Tiktoken or Transformers tokenizer); or a token counter that returns the number of tokens in an input.

`chunk_size` is the maximum number of tokens a chunk may contain. It defaults to `None`, in which case it will be set to the same value as the tokenizer's `model_max_length` attribute (minus the number of tokens returned by attempting to tokenize an empty string) if possible, otherwise a `ValueError` will be raised.

`chunking_model` is the name of the [Isaacus enrichment model](https://docs.isaacus.com/models/introduction#enrichment) to use for AI chunking. This argument defaults to `None`, in which case, unless you provide an [Isaacus Legal Graph Schema (ILGS) Document](https://docs.isaacus.com/ilgs/introduction) as input, AI chunking will be disabled.

`isaacus_client` is an instance of the `isaacus.Isaacus` API client to use for AI chunking instead of a client constructed with default parameters. This argument defaults to `None`, in which case a client will be constructed with default parameters if AI chunking is enabled.

`tokenizer_kwargs` is an optional dictionary of keyword arguments to be passed to the tokenizer or token counter whenever it is called. This can be used to disable the current default behavior of treating any encountered special tokens as if they are normal text when using a Tiktoken or Transformers tokenizer. This argument defaults to `None`, in which case no additional keyword arguments will be passed to the tokenizer or token counter.

`max_token_chars` is the maximum number of characters a token may contain. It is used to significantly speed up the token counting of long inputs. It defaults to `None` in which case it will either not be used or will, if possible, be set to the number of characters in the longest token in the tokenizer's vocabulary as determined by the `token_byte_values` or `get_vocab` methods.

`memoize` flags whether to memoize the token counter. It defaults to `True`.

`cache_maxsize` is the maximum number of text-token count pairs that can be stored in the token counter's cache. It defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

This function returns a chunker that takes either a single input or a sequence of inputs and returns, depending on whether multiple inputs have been provided, a list or list of lists of chunks up to `chunk_size`-tokens-long with the whitespace used to split the input removed, and, if the optional `offsets` argument to the chunker is `True`, a list or lists of tuples of the form `(start, end)` where `start` is the index of the first character of a chunk in an input and `end` is the index of the character succeeding the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

The resulting chunker can be passed a `processes` argument that specifies the number of processes to be used when chunking multiple inputs.

It is also possible to pass a `progress` argument which, if set to `True` and multiple inputs are passed, will display a progress bar.

As described above, the `offsets` argument, if set to `True`, will cause the chunker to return the start and end offsets of each chunk.

The chunker accepts an `overlap` argument that specifies the proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. It defaults to `None`, in which case no overlapping occurs.

### `chunk()`

```python
def chunk(
    text: str | ILGSDocument,
    chunk_size: int,
    token_counter: Callable[[str], int],
    *,
    memoize: bool = True,
    offsets: bool = False,
    overlap: float | int | None = None,
    chunking_model: str | None = None,
    isaacus_client: "isaacus.Isaacus | None" = None,
    cache_maxsize: int | None = None,
) -> list[str] | tuple[list[str], list[tuple[int, int]]]:
```

`chunk()` splits a text into semantically meaningful chunks of a specified size as determined by the provided token counter.

`text` is the input to be chunked. If you pass an Isaacus Legal Graph Schema (ILGS) Document, AI chunking will occur automatically without re-enriching the document.

`chunk_size` is the maximum number of tokens a chunk may contain.

`token_counter` is a callable that takes a string and returns the number of tokens in it.

`memoize` flags whether to memoize the token counter. It defaults to `True`.

`offsets` flags whether to return the start and end offsets of each chunk. It defaults to `False`.

`overlap` specifies the proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. It defaults to `None`, in which case no overlapping occurs.

`chunking_model` is the name of the [Isaacus enrichment model](https://docs.isaacus.com/models/introduction#enrichment) to use for AI chunking. This argument defaults to `None`, in which case, unless you provide an Isaacus Legal Graph Schema (ILGS) Document as input, AI chunking will be disabled.

`isaacus_client` is an instance of the `isaacus.Isaacus` API client to use for AI chunking instead of a client constructed with default parameters. This argument defaults to `None`, in which case a client will be constructed with default parameters if AI chunking is enabled.

`cache_maxsize` is the maximum number of text-token count pairs that can be stored in the token counter's cache. It defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

This function returns a list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed, and, if `offsets` is `True`, a list of tuples of the form `(start, end)` where `start` is the index of the first character of the chunk in the original text and `end` is the index of the character after the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

## How It Works 🔍

semchunk works by recursively splitting texts until all resulting chunks are less than or equal to a specified chunk size.

In particular, it:

1. splits text using the most structurally meaningful splitter possible;
2. recursively splits the resulting chunks until a set of chunks less than or equal to the specified chunk size is produced;
3. merges any chunks that are under the chunk size back together until the chunk size is reached;
4. reattaches any non-whitespace splitters back to the ends of chunks except for the last chunk if doing so does not bring chunks over the chunk size, otherwise adds non-whitespace splitters as their own chunks; and
5. since version 3.0.0, excludes chunks consisting entirely of whitespace characters.

To ensure that chunks are as semantically meaningful as possible, semchunk uses the following splitters, in order of precedence:

1. the largest sequence of newlines (`\n`) and/or carriage returns (`\r`);
2. the largest sequence of tabs;
3. the largest sequence of whitespace characters (as defined by regex's `\s` character class) or, since version 3.2.0, if the largest sequence of whitespace characters is only a single character and there exist whitespace characters preceded by any of the structurally meaningful non-whitespace characters listed below (in the same order of precedence), then only those specific whitespace characters;
4. sentence terminators (`.`, `?`, and `!`);
5. clause separators (`;`, `,`, `*`, `(`, `)`, `[`, `]`, `“`, `”`, `‘`, `’`, `'`, `"`, `` ` ``);
6. sentence interrupters (`:`, `—` and `…`);
7. word joiners (`/`, `\`, `–`, `&` and `-`); and
8. all other characters.

Where AI-powered chunking is enabled, semchunk:

1. splits text into chunks up to 1,000,000-characters-long using the above algorithm in order to avoid sending excessively long inputs to an enrichment model;
2. enriches the resulting chunks with the enrichment model, pooling all unique spans extracted from each enriched chunk together;
3. for each level of depth, creates new spans where necessary to ensure that all content at that level of depth is covered by a span;
4. constructs a tree of spans based on containment;
5. iterates through the span tree:
   1. adding spans to chunks until the chunk size is reached;
   2. discarding whitespace-only chunks;
   3. removing leading and trailing whitespace from chunks;
   4. entering into the children of spans where a span exceeds the chunk size; and
   5. falling back to the above algorithm where a span has no children.

If overlapping chunks have been requested, semchunk also:

1. internally reduces the chunk size to `min(overlap, chunk_size - overlap)` (`overlap` being computed as `floor(chunk_size * overlap)` for relative overlaps and `min(overlap, chunk_size - 1)` for absolute overlaps); and
2. merges every `floor(original_chunk_size / reduced_chunk_size)` chunks starting from the first chunk and then jumping by `floor((original_chunk_size - overlap) / reduced_chunk_size)` chunks until the last chunk is reached.

## Citation 📝

If you use semchunk for research, please cite it as follows:

```bibtex
@software{butler2023semchunk,
  author       = {Butler, Umar},
  title        = {semchunk: a Python library for semantic chunking},
  year         = {2023},
  url          = {https://github.com/isaacus-dev/semchunk},
  version      = {4.0.0},
  publisher    = {Isaacus}
}
```

## License 📜

This library is licensed under the [MIT License](https://github.com/isaacus-dev/semchunk/blob/main/LICENCE).

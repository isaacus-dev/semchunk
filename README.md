<div align='center'>

# semchunk 🧩
<a href="https://pypi.org/project/semchunk/" alt="PyPI Version"><img src="https://img.shields.io/pypi/v/semchunk"></a> <a href="https://github.com/isaacus-dev/semchunk/actions/workflows/ci.yml" alt="Build Status"><img src="https://img.shields.io/github/actions/workflow/status/isaacus-dev/semchunk/ci.yml?branch=main"></a> <a href="https://app.codecov.io/gh/isaacus-dev/semchunk" alt="Code Coverage"><img src="https://img.shields.io/codecov/c/github/isaacus-dev/semchunk"></a> <a href="https://pepy.tech/project/semchunk" alt="Downloads"><img src="https://pepy.tech/badge/semchunk/month"></a>

</div>

**`semchunk`** by [**Isaacus**](https://isaacus.com/) is a fast, lightweight and easy-to-use Python library for splitting text into semantically meaningful chunks.

It has built-in support for tokenizers from OpenAI's `tiktoken` and Hugging Face's `transformers` and `tokenizers` libraries, in addition to supporting custom tokenizers and token counters. It can also overlap chunks as well as return their offsets.

Powered by an efficient yet highly accurate chunking algorithm ([How It Works 🔍](https://github.com/isaacus-dev/semchunk#how-it-works-)), `semchunk` produces chunks that are more semantically meaningful than regular token and recursive character chunkers like `langchain`'s `RecursiveCharacterTextSplitter`, while also being 85% faster than its closest alternative, `semantic-text-splitter` ([Benchmarks 📊](https://github.com/isaacus-dev/semchunk#benchmarks-)).

`semchunk` is production ready, being used every day in the [Isaacus API](https://docs.isaacus.com) to split extremely long legal documents into more manageable chunks for our [Kanon legal AI models](https://docs.isaacus.com/models).

## Installation 📦
`semchunk` can be installed with `pip`:
```bash
pip install semchunk
```

`semchunk` is also available on `conda-forge`:
```bash
conda install conda-forge::semchunk
# or
conda install -c conda-forge semchunk
```

In addition, [@dominictarro](https://github.com/dominictarro) maintains a Rust port of `semchunk` named [`semchunk-rs`](https://crates.io/crates/semchunk-rs).

## Quickstart 👩‍💻
The code snippet below demonstrates how to chunk text with `semchunk`:
```python
import semchunk
import tiktoken                        # `transformers` and `tiktoken` are not required.
from transformers import AutoTokenizer # They're just here for demonstration purposes.

chunk_size = 4 # A low chunk size is used here for demonstration purposes. Keep in mind, `semchunk`
               # does not know how many special tokens, if any, your tokenizer adds to every input,
               # so you may want to deduct the number of special tokens added from your chunk size.
text = 'The quick brown fox jumps over the lazy dog.'

# You can construct a chunker with `semchunk.chunkerify()` by passing the name of an OpenAI model,
# OpenAI `tiktoken` encoding or Hugging Face model, or a custom tokenizer that has an `encode()`
# method (like a `tiktoken`, `transformers` or `tokenizers` tokenizer) or a custom token counting
# function that takes a text and returns the number of tokens in it.
chunker = semchunk.chunkerify('isaacus/kanon-tokenizer', chunk_size) or \
          semchunk.chunkerify('gpt-4', chunk_size) or \
          semchunk.chunkerify('cl100k_base', chunk_size) or \
          semchunk.chunkerify(AutoTokenizer.from_pretrained('isaacus/kanon-tokenizer'), chunk_size) or \
          semchunk.chunkerify(tiktoken.encoding_for_model('gpt-4'), chunk_size) or \
          semchunk.chunkerify(lambda text: len(text.split()), chunk_size)

# If you give the resulting chunker a single text, it'll return a list of chunks. If you give it a
# list of texts, it'll return a list of lists of chunks.
assert chunker(text) == ['The quick brown fox', 'jumps over the', 'lazy dog.']
assert chunker([text], progress = True) == [['The quick brown fox', 'jumps over the', 'lazy dog.']]

# If you have a lot of texts and you want to speed things up, you can enable multiprocessing by
# setting `processes` to a number greater than 1.
assert chunker([text], processes = 2) == [['The quick brown fox', 'jumps over the', 'lazy dog.']]

# You can also pass an `offsets` argument to return the offsets of chunks, as well as an `overlap`
# argument to overlap chunks by a ratio (if < 1) or an absolute number of tokens (if >= 1).
chunks, offsets = chunker(text, offsets = True, overlap = 0.5)
```

## Usage 🕹️
### `chunkerify()`
```python
def chunkerify(
    tokenizer_or_token_counter: str | tiktoken.Encoding | transformers.PreTrainedTokenizer | \
                                tokenizers.Tokenizer | Callable[[str], int],
    chunk_size: int = None,
    max_token_chars: int = None,
    memoize: bool = True,
    cache_maxsize: int | None = None,
) -> Callable[[str | Sequence[str], bool, bool, bool, int | float | None], list[str] | tuple[list[str], list[tuple[int, int]]] | list[list[str]] | tuple[list[list[str]], list[list[tuple[int, int]]]]]:
```

`chunkerify()` constructs a chunker that splits one or more texts into semantically meaningful chunks of a specified size as determined by the provided tokenizer or token counter.

`tokenizer_or_token_counter` is either: the name of a `tiktoken` or `transformers` tokenizer (with priority given to the former); a tokenizer that possesses an `encode` attribute (e.g., a `tiktoken`, `transformers` or `tokenizers` tokenizer); or a token counter that returns the number of tokens in an input.

`chunk_size` is the maximum number of tokens a chunk may contain. It defaults to `None` in which case it will be set to the same value as the tokenizer's `model_max_length` attribute (deducted by the number of tokens returned by attempting to tokenize an empty string) if possible, otherwise a `ValueError` will be raised.

`max_token_chars` is the maximum number of characters a token may contain. It is used to significantly speed up the token counting of long inputs. It defaults to `None` in which case it will either not be used or will, if possible, be set to the number of characters in the longest token in the tokenizer's vocabulary as determined by the `token_byte_values` or `get_vocab` methods.

`memoize` flags whether to memoize the token counter. It defaults to `True`.

`cache_maxsize` is the maximum number of text-token count pairs that can be stored in the token counter's cache. It defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

This function returns a chunker that takes either a single text or a sequence of texts and returns, depending on whether multiple texts have been provided, a list or list of lists of chunks up to `chunk_size`-tokens-long with any whitespace used to split the text removed, and, if the optional `offsets` argument to the chunker is `True`, a list or lists of tuples of the form `(start, end)` where `start` is the index of the first character of a chunk in a text and `end` is the index of the character succeeding the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

The resulting chunker can be passed a `processes` argument that specifies the number of processes to be used when chunking multiple texts.

It is also possible to pass a `progress` argument which, if set to `True` and multiple texts are passed, will display a progress bar.

As described above, the `offsets` argument, if set to `True`, will cause the chunker to return the start and end offsets of each chunk.

The chunker accepts an `overlap` argument that specifies the proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. It defaults to `None`, in which case no overlapping occurs.

### `chunk()`
```python
def chunk(
    text: str,
    chunk_size: int,
    token_counter: Callable,
    memoize: bool = True,
    offsets: bool = False,
    overlap: float | int | None = None,
    cache_maxsize: int | None = None,
) -> list[str]
```

`chunk()` splits a text into semantically meaningful chunks of a specified size as determined by the provided token counter.

`text` is the text to be chunked.

`chunk_size` is the maximum number of tokens a chunk may contain.

`token_counter` is a callable that takes a string and returns the number of tokens in it.

`memoize` flags whether to memoize the token counter. It defaults to `True`.

`offsets` flags whether to return the start and end offsets of each chunk. It defaults to `False`.

`overlap` specifies the proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap. It defaults to `None`, in which case no overlapping occurs.

`cache_maxsize` is the maximum number of text-token count pairs that can be stored in the token counter's cache. It defaults to `None`, which makes the cache unbounded. This argument is only used if `memoize` is `True`.

This function returns a list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed, and, if `offsets` is `True`, a list of tuples of the form `(start, end)` where `start` is the index of the first character of the chunk in the original text and `end` is the index of the character after the last character of the chunk such that `chunks[i] == text[offsets[i][0]:offsets[i][1]]`.

## How It Works 🔍
`semchunk` works by recursively splitting texts until all resulting chunks are equal to or less than a specified chunk size. In particular, it:
1. Splits text using the most semantically meaningful splitter possible;
1. Recursively splits the resulting chunks until a set of chunks equal to or less than the specified chunk size is produced;
1. Merges any chunks that are under the chunk size back together until the chunk size is reached;
1. Reattaches any non-whitespace splitters back to the ends of chunks barring the final chunk if doing so does not bring chunks over the chunk size, otherwise adds non-whitespace splitters as their own chunks; and
1. Since version 3.0.0, excludes chunks consisting entirely of whitespace characters.

To ensure that chunks are as semantically meaningful as possible, `semchunk` uses the following splitters, in order of precedence:
1. The largest sequence of newlines (`\n`) and/or carriage returns (`\r`);
1. The largest sequence of tabs;
1. The largest sequence of whitespace characters (as defined by regex's `\s` character class) or, since version 3.2.0, if the largest sequence of whitespace characters is only a single character and there exist whitespace characters preceded by any of the semantically meaningful non-whitespace characters listed below (in the same order of precedence), then only those specific whitespace characters;
1. Sentence terminators (`.`, `?`, `!` and `*`);
1. Clause separators (`;`, `,`, `(`, `)`, `[`, `]`, `“`, `”`, `‘`, `’`, `'`, `"` and `` ` ``);
1. Sentence interrupters (`:`, `—` and `…`);
1. Word joiners (`/`, `\`, `–`, `&` and `-`); and
1. All other characters.

If overlapping chunks have been requested, `semchunk` also:
1. Internally reduces the chunk size to `min(overlap, chunk_size - overlap)` (`overlap` being computed as `floor(chunk_size * overlap)` for relative overlaps and `min(overlap, chunk_size - 1)` for absolute overlaps); and
1. Merges every `floor(original_chunk_size / reduced_chunk_size)` chunks starting from the first chunk and then jumping by `floor((original_chunk_size - overlap) / reduced_chunk_size)` chunks until the last chunk is reached.

## Benchmarks 📊
On a desktop with a Ryzen 9 7900X, 96 GB of DDR5 5600MHz CL40 RAM, Windows 11 and Python 3.12.4, it takes `semchunk` 3.04 seconds to split every sample in [NLTK's Gutenberg Corpus](https://www.nltk.org/howto/corpus.html#plaintext-corpora) into 512-token-long chunks with GPT-4's tokenizer (for context, the Corpus contains 18 texts and 3,001,260 tokens). By comparison, it takes [`semantic-text-splitter`](https://pypi.org/project/semantic-text-splitter/) (with multiprocessing) 24.84 seconds to chunk the same texts into 512-token-long chunks — a difference of 87.76%.

The code used to benchmark `semchunk` and `semantic-text-splitter` is available [here](https://github.com/isaacus-dev/semchunk/blob/main/tests/bench.py).

## Licence 📄
This library is licensed under the [MIT License](https://github.com/isaacus-dev/semchunk/blob/main/LICENCE).
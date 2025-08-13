## Changelog 🔄
All notable changes to `semchunk` will be documented here. This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.3] - 2025-08-13
### Fixed
- Fixed broken Python download count shield ([crflynn/pypistats.org#82](https://github.com/crflynn/pypistats.org/issues/82#issue-3285911460)).

## [3.2.2] - 2025-06-09
### Fixed
- Fixed `IndexError` being raised when chunking whitespace only texts with overlapping enabled ([#18](https://github.com/isaacus-dev/semchunk/issues/18)).

## [3.2.1] - 2025-03-27
### Fixed
- Fixed minor typos in the README and docstrings.

## [3.2.0] - 2025-03-20
### Changed
- Significantly improved the quality of chunks produced when chunking with low chunk sizes or documents with minimal varying levels of whitespace by adding a new rule to the `semchunk` algorithm that prioritizes splitting at the occurrence of single whitespace characters preceded by hierarchically meaningful non-whitespace characters over splitting at all single whitespace characters in general ([#17](https://github.com/isaacus-dev/semchunk/issues/17)).

## [3.1.3] - 2025-03-10
### Changed
- Added mention of Isaacus to the README.

## [3.1.2] - 2025-03-06
### Changed
- Changed test model from `isaacus/emubert` to `isaacus/kanon-tokenizer`.

## [3.1.1] - 2025-02-18
### Added
- Added a note to the quickstart section of the README advising users to deduct the number of special tokens automatically added by their tokenizer from their chunk size. This note had been removed in version 3.0.0 but has been readded as it is unlikely to be obvious to users.

## [3.1.0] - 2025-02-16
### Added
- Introduced a new `cache_maxsize` argument to `chunkerify()` and `chunk()` that specifies the maximum number of text-token count pairs that can be stored in a token counter's cache. The argument defaults to `None`, in which case the cache is unbounded.

## [3.0.4] - 2025-02-14
### Fixed
- Fixed bug where attempting to chunk only whitespace characters would raise `ValueError: not enough values to unpack (expected 2, got 0)` ([ScrapeGraphAI/Scrapegraph-ai#893](https://github.com/ScrapeGraphAI/Scrapegraph-ai/issues/893)).

## [3.0.3] - 2025-02-13
### Fixed
- Fixed `isaacus/emubert` mistakenly being set to `isaacus-dev/emubert` in the README and tests.

## [3.0.2] - 2025-02-13
### Fixed
- Significantly sped up chunking very long texts with little to no variation in levels of whitespace used (fixing [#8](https://github.com/isaacus-dev/semchunk/issues/8)) and, in the process, also slightly improved overall performance.

### Changed
- Transferred `semchunk` to [Isaacus](https://isaacus.com/).
- Began formatting with Ruff.

## [3.0.1] - 2024-01-10
### Fixed
- Fixed a bug where attempting to chunk an empty text would raise a `ValueError`.

## [3.0.0] - 2024-12-31
### Added
- Added an `offsets` argument to `chunk()` and `Chunker.__call__()` that specifies whether to return the start and end offsets of each chunk ([#9](https://github.com/isaacus-dev/semchunk/issues/9)). The argument defaults to `False`.
- Added an `overlap` argument to `chunk()` and `Chunker.__call__()` that specifies the proportion of the chunk size, or, if >=1, the number of tokens, by which chunks should overlap ([#1](https://github.com/isaacus-dev/semchunk/issues/1)). The argument defaults to `None`, in which case no overlapping occurs.
- Added an undocumented, private `_make_chunk_function()` method to the `Chunker` class that constructs chunking functions with call-level arguments passed.
- Added more unit tests for new features as well as for multiple token counters and for ensuring there are no chunks comprised entirely of whitespace characters.

### Changed
- Began removing chunks comprised entirely of whitespace characters from the output of `chunk()`.
- Updated `semchunk`'s description from 'A fast and lightweight Python library for splitting text into semantically meaningful chunks.' and 'A fast, lightweight and easy-to-use Python library for splitting text into semantically meaningful chunks.'.

### Fixed
- Fixed a typo in the docstring for the `__call__()` method of the `Chunker` class returned by `chunkerify()` where most of the documentation for the arguments were listed under the section for the method's returns.

### Removed
- Removed undocumented, private `chunk()` method from the `Chunker` class returned by `chunkerify()`.
- Removed undocumented, private `_reattach_whitespace_splitters` argument of `chunk()` that was introduced to experiment with potentially adding support for overlap ratios.

## [2.2.2] - 2024-12-18
### Fixed
- Ensured `hatch` does not include irrelevant files in the distribution.

## [2.2.1] - 2024-12-17
### Changed
- Started benchmarking [`semantic-text-splitter`](https://pypi.org/project/semantic-text-splitter/) in parallel to ensure a fair comparison, courtesy of [@benbrandt](https://github.com/benbrandt) ([#17](https://github.com/isaacus-dev/semchunk/pull/12)).

## [2.2.0] - 2024-07-12
### Changed
- Switched from having `chunkerify()` output a function to having it return an instance of the new `Chunker()` class which should not alter functionality in any way but will allow for the preservation of type hints, fixing [#7](https://github.com/isaacus-dev/semchunk/pull/7).

## [2.1.0] - 2024-06-20
### Fixed
- Ceased memoizing `chunk()` (but not token counters) due to the fact that cached outputs of memoized functions are shallow rather than deep copies of original outputs, meaning that if one were to chunk a text and then chunk that same text again and then modify one of the chunks outputted by the first call, the chunks outputted by the second call would also be modified. This behaviour is not expected and therefore undesirable. The memoization of token counters is not impacted as they output immutable objects, namely, integers.

## [2.0.0] - 2024-06-19
### Added
- Added support for multiprocessing through the `processes` argument passable to chunkers constructed by `chunkerify()`.

### Removed
- No longer guaranteed that `semchunk` is pure Python.

## [1.0.1] - 2024-06-02
### Fixed
- Documented the `progress` argument in the docstring for `chunkerify()` and its type hint in the README.

## [1.0.0] - 2024-06-02
### Added
- Added a `progress` argument to the chunker returned by `chunkerify()` that, when set to `True` and multiple texts are passed, displays a progress bar.

## [0.3.2] - 2024-06-01
### Fixed
- Fixed a bug where a `DivisionByZeroError` would be raised where a token counter returned zero tokens when called from `merge_splits()`, courtesy of [@jcobol](https://github.com/jcobol) ([#5](https://github.com/isaacus-dev/semchunk/pull/5)) ([7fd64eb](https://github.com/isaacus-dev/semchunk/pull/5/commits/7fd64eb8cf51f45702c59f43795be9a00c7d0d17)), fixing [#4](https://github.com/isaacus-dev/semchunk/issues/4).

## [0.3.1] - 2024-05-18
### Fixed
- Fixed typo in error messages in `chunkerify()` where it was referred to as `make_chunker()`.

## [0.3.0] - 2024-05-18
### Added
- Introduced the `chunkerify()` function, which constructs a chunker from a tokenizer or token counter that can be reused and can also chunk multiple texts in a single call. The resulting chunker speeds up chunking by 40.4% thanks, in large part, to a token counter that avoid having to count the number of tokens in a text when the number of characters in the text exceed a certain threshold, courtesy of [@R0bk](https://github.com/R0bk) ([#3](https://github.com/isaacus-dev/semchunk/pull/3)) ([337a186](https://github.com/isaacus-dev/semchunk/pull/3/commits/337a18615f991076b076262288b0408cb162b48c)).

## [0.2.4] - 2024-05-13
### Changed
- Improved chunking performance with larger chunk sizes by switching from linear to binary search for the identification of optimal chunk boundaries, courtesy of [@R0bk](https://github.com/R0bk) ([#3](https://github.com/isaacus-dev/semchunk/pull/3)) ([337a186](https://github.com/isaacus-dev/semchunk/pull/3/commits/337a18615f991076b076262288b0408cb162b48c)).

## [0.2.3] - 2024-03-11
### Fixed
- Ensured that memoization does not overwrite `chunk()`'s function signature.

## [0.2.2] - 2024-02-05
### Fixed
- Ensured that the `memoize` argument is passed back to `chunk()` in recursive calls.

## [0.2.1] - 2023-11-09
### Added
- Memoized `chunk()`.

### Fixed
- Fixed typos in README.

## [0.2.0] - 2023-11-07
### Added
- Added the `memoize` argument to `chunk()`, which memoizes token counters by default to significantly improve performance.

### Changed
- Improved chunking performance.

## [0.1.2] - 2023-11-07
### Fixed
- Fixed links in the README.

## [0.1.1] - 2023-11-07
### Added
- Added new test samples.
- Added benchmarks.

### Changed
- Improved chunking performance.
- improved test coverage.

## [0.1.0] - 2023-11-05
### Added
- Added the `chunk()` function, which splits text into semantically meaningful chunks of a specified size as determined by a provided token counter.

[3.2.2]: https://github.com/isaacus-dev/semchunk/compare/v3.2.1...v3.2.2
[3.2.1]: https://github.com/isaacus-dev/semchunk/compare/v3.2.0...v3.2.1
[3.2.0]: https://github.com/isaacus-dev/semchunk/compare/v3.1.3...v3.2.0
[3.1.3]: https://github.com/isaacus-dev/semchunk/compare/v3.1.2...v3.1.3
[3.1.2]: https://github.com/isaacus-dev/semchunk/compare/v3.1.1...v3.1.2
[3.1.1]: https://github.com/isaacus-dev/semchunk/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/isaacus-dev/semchunk/compare/v3.0.4...v3.1.0
[3.0.4]: https://github.com/isaacus-dev/semchunk/compare/v3.0.3...v3.0.4
[3.0.3]: https://github.com/isaacus-dev/semchunk/compare/v3.0.2...v3.0.3
[3.0.2]: https://github.com/isaacus-dev/semchunk/compare/v3.0.1...v3.0.2
[3.0.1]: https://github.com/isaacus-dev/semchunk/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/isaacus-dev/semchunk/compare/v2.2.2...v3.0.0
[2.2.2]: https://github.com/isaacus-dev/semchunk/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/isaacus-dev/semchunk/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/isaacus-dev/semchunk/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/isaacus-dev/semchunk/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/isaacus-dev/semchunk/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/isaacus-dev/semchunk/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/isaacus-dev/semchunk/compare/v0.3.2...v1.0.0
[0.3.2]: https://github.com/isaacus-dev/semchunk/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/isaacus-dev/semchunk/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/isaacus-dev/semchunk/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/isaacus-dev/semchunk/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/isaacus-dev/semchunk/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/isaacus-dev/semchunk/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/isaacus-dev/semchunk/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/isaacus-dev/semchunk/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/isaacus-dev/semchunk/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/isaacus-dev/semchunk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/isaacus-dev/semchunk/releases/tag/v0.1.0
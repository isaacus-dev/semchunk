"""Test semchunk."""
from __future__ import annotations

import math

import semchunk

import tiktoken

from helpers import GUTENBERG, initialize_test_token_counters
from transformers import AutoTokenizer

TEST_TOKEN_COUNTERS = (
    # 'emubert_transformers',
    # 'gpt4_tiktoken',
    # 'word',
    'char',
)
TEST_CHUNK_SIZES = (
    10,
    512,
)
TEST_OFFSETS = True

DETERMINISTIC_TEST_INPUT = 'ThisIs\tATest.'
DETERMINISTIC_TEST_CHUNK_SIZE = 4
DETERMINISTIC_TEST_OUTPUT_CHUNKS = {
    'gpt4_tiktoken': ['ThisIs', 'ATest.'],
    'emubert_transformers': ['ThisIs', 'ATest.'],
    'word': ['ThisIs\tATest.'],
    'char': ['This', 'Is', 'ATes', 't.'],
}
DETERMINISTIC_TEST_OUTPUT_OFFSETS = {
    'gpt4_tiktoken': [(0, 6), (7, 13)],
    'emubert_transformers': [(0, 6), (7, 13)],
    'word': [(0, 13)],
    'char': [(0, 4), (4, 6), (7, 11), (11, 13)],
}

def test_semchunk() -> None:
    """Test semchunk."""

    # Initalize test token counters.
    token_counters = initialize_test_token_counters()
    token_counters = {name: token_counters[name] for name in TEST_TOKEN_COUNTERS}
    
    # Test chunking with the token counters.
    for name, token_counter in token_counters.items():
        print(f'Testing {name}...')
        
        # Test chunking with a variety of chunk sizes.
        for chunk_size in TEST_CHUNK_SIZES:
            # Add the number of special tokens added by the tokenizer to the chunk size + 1 if the tokenizer adds special tokens.
            if token_counter(''):
                chunk_size += token_counter('') + 1
            
            # Test chunking with a variety of texts.
            for fileid in GUTENBERG.fileids():
                sample = GUTENBERG.raw(fileid)
                print(f'Chunking {fileid} with chunk size {chunk_size}...')
                
                chunker = semchunk.chunkerify(token_counter, chunk_size)
                chunks = chunker(sample)
                
                for chunk in chunks:
                    assert token_counter(chunk) <= chunk_size
                    assert chunk and not chunk.isspace()
                
                if TEST_OFFSETS:
                    chunks, offsets = chunker(sample, offsets = True)
                    
                    for chunk, (start, end) in zip(chunks, offsets):
                        assert token_counter(chunk) <= chunk_size
                        assert chunk == sample[start:end]
                        assert chunk and not chunk.isspace()
                
                # Verify that recombining lowercased chunks stripped of whitespace yields the original text.
                lowercased_no_whitespace = ''.join(sample.lower().split())

                if TEST_OFFSETS:
                    chunks, offsets = chunker(lowercased_no_whitespace, offsets = True)
                    assert ''.join(chunks) == lowercased_no_whitespace
                    assert ''.join(lowercased_no_whitespace[start:end] for start, end in offsets) == lowercased_no_whitespace
                    
                chunks = chunker(lowercased_no_whitespace)
                assert ''.join(chunks) == lowercased_no_whitespace
        
        # Test overlapping.
        chunker = semchunk.chunkerify(token_counter, DETERMINISTIC_TEST_CHUNK_SIZE)
        low_overlap_chunks = chunker(DETERMINISTIC_TEST_INPUT, overlap = 0.1)
        high_overlap_chunks = chunker(DETERMINISTIC_TEST_INPUT, overlap = math.ceil(DETERMINISTIC_TEST_CHUNK_SIZE * 0.9))
        
        if name == 'word':
            assert len(high_overlap_chunks) == len(low_overlap_chunks)
        
        else:
            assert len(high_overlap_chunks) > len(low_overlap_chunks)
        
        if TEST_OFFSETS:
            low_overlap_chunks, low_overlap_offsets = chunker(DETERMINISTIC_TEST_INPUT, overlap = 0.1, offsets = True)
            high_overlap_chunks, high_overlap_offsets = chunker(DETERMINISTIC_TEST_INPUT, overlap = math.ceil(DETERMINISTIC_TEST_CHUNK_SIZE * 0.9), offsets = True)
            
            if name == 'word':
                assert len(high_overlap_chunks) == len(low_overlap_chunks)
                assert len(high_overlap_offsets) == len(low_overlap_offsets)

            else:
                assert len(high_overlap_chunks) > len(low_overlap_chunks)
                assert len(high_overlap_offsets) > len(low_overlap_offsets)
    
            assert high_overlap_chunks == [DETERMINISTIC_TEST_INPUT[start:end] for start, end in high_overlap_offsets]
        
        # Verify deterministic behavior.
        chunker = semchunk.chunkerify(token_counter, DETERMINISTIC_TEST_CHUNK_SIZE)
        chunks = chunker(DETERMINISTIC_TEST_INPUT)
        assert chunks == DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]
        
        if TEST_OFFSETS:
            chunks, offsets = chunker(DETERMINISTIC_TEST_INPUT, offsets = True)
            assert chunks == DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]
            assert offsets == DETERMINISTIC_TEST_OUTPUT_OFFSETS[name]
            
        # Test using semchunk directly with memoization enabled.
        chunks = semchunk.chunk(DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_CHUNK_SIZE, token_counter, memoize = True)
        assert chunks == DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]
        
        if TEST_OFFSETS:
            chunks, offsets = semchunk.chunk(DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_CHUNK_SIZE, token_counter, offsets = True, memoize = True)
            assert chunks == DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]
            assert offsets == DETERMINISTIC_TEST_OUTPUT_OFFSETS[name]
        
        # Test chunking multiple texts.
        chunker = semchunk.chunkerify(token_counter, DETERMINISTIC_TEST_CHUNK_SIZE)
        chunks = chunker([DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_INPUT])
        assert chunks == [DETERMINISTIC_TEST_OUTPUT_CHUNKS[name], DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]]
        
        if TEST_OFFSETS:
            chunks, offsets = chunker([DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_INPUT], offsets = True)
            assert chunks == [DETERMINISTIC_TEST_OUTPUT_CHUNKS[name], DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]]
        
        # Test chunking multiple texts with multiple processes.
        chunker = semchunk.chunkerify(token_counter, DETERMINISTIC_TEST_CHUNK_SIZE)
        chunks = chunker([DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_INPUT], processes = 2)
        assert chunks == [DETERMINISTIC_TEST_OUTPUT_CHUNKS[name], DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]]
        
        if TEST_OFFSETS:
            chunks, offsets = chunker([DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_INPUT], offsets = True, processes = 2)
            assert chunks == [DETERMINISTIC_TEST_OUTPUT_CHUNKS[name], DETERMINISTIC_TEST_OUTPUT_CHUNKS[name]]
            assert offsets == [DETERMINISTIC_TEST_OUTPUT_OFFSETS[name], DETERMINISTIC_TEST_OUTPUT_OFFSETS[name]]
    
    # Test causing a `ValueError` by passing a token counter without a chunk size.
    try:
        chunker = semchunk.chunkerify(list(token_counters.values())[0], None)
        error_raised = False
    
    except ValueError:
        error_raised = True
    
    assert error_raised
    
    # Test using `tiktoken` tokenizers, encodings and a `transformers` tokenizer by name with `chunkerify()`.
    for name in ['cl100k_base', 'gpt-4', 'isaacus/kanon-tokenizer']:
        chunker = semchunk.chunkerify(name, 1)
        chunker(DETERMINISTIC_TEST_INPUT)
        if TEST_OFFSETS: chunker(DETERMINISTIC_TEST_INPUT, offsets = True)

    # Test causing a `ValueError` by passing a tokenizer by name that should not exist.
    try:
        chunker = semchunk.chunkerify('\n\f\rع\n\f\r', 1)
        error_raised = False
    
    except ValueError:
        error_raised = True
    
    assert error_raised
    
    # Test using a `transformers` tokenizer directly.
    tokenizer = AutoTokenizer.from_pretrained('isaacus/kanon-tokenizer')
    chunker = semchunk.chunkerify(tokenizer, 1)
    
    # Test using a `tiktoken` tokenizer directly.
    tokenizer = tiktoken.encoding_for_model('gpt-4')
    chunker = semchunk.chunkerify(tokenizer, 1)
    
    # Try enabling a progress bar.
    chunker([DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_INPUT], progress = True)
    chunker([DETERMINISTIC_TEST_INPUT, DETERMINISTIC_TEST_INPUT], offsets = True, progress = True)
    
    # Test chunking nothing to ensure no errors are raised.
    semchunk.chunk('', 512, lambda *args: 0)
    
    # Test chunking whitespace to ensure no errors are raised.
    semchunk.chunk('\n\n', 512, lambda *args: 0)

if __name__ == '__main__':
    test_semchunk()
from __future__ import annotations

from typing import Callable

import nltk
import tiktoken
import transformers

try:
    GUTENBERG = nltk.corpus.gutenberg
    GUTENBERG.fileids()
    
except LookupError:
    nltk.download('gutenberg')

def tokenizer_to_token_counter(tokenizer: Callable[[str], list[str | int]]) -> Callable[[str], int]:
    """Convert a tokenizer into a token counter."""
    
    def token_counter(text: str) -> int:
        """Count the number of tokens in a text."""
        
        return len(tokenizer(text))
    
    return token_counter

def make_transformers_tokenizer(tokenizer: Callable[[str], list[int]]) -> Callable[[str], int]:
    """Convert a `transformers` tokenizer into a tokenizer function."""
    
    def transformers_tokenizer(text: str) -> list[int]:
        """Tokenize a text using a `transformers` tokenizer."""
        
        return tokenizer.encode(text, add_special_tokens = False)
    
    return transformers_tokenizer

def initialize_test_token_counters() -> dict[str, Callable[[str], int]]:
    """Initialize `tiktoken`, `transformers`, character and word token counters for testing purposes."""
    
    gpt4_tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-4').encode
    emubert_transformers_tokenizer = make_transformers_tokenizer(transformers.AutoTokenizer.from_pretrained('isaacus/kanon-tokenizer'))
    
    def word_tokenizer(text: str) -> list[str]:
        """Tokenize a text into words."""
        
        return text.split()
    
    tokenizers = {
        'gpt4_tiktoken': gpt4_tiktoken_tokenizer,
        'emubert_transformers': emubert_transformers_tokenizer,
        'word': word_tokenizer,
        'char': list,
    }
    
    return {name: tokenizer_to_token_counter(tokenizer) for name, tokenizer in tokenizers.items()}
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Token Estimation Utilities for Language Models.

This module provides improved token estimation algorithms that are more accurate
than simple character-based counting. The estimations are designed to work
without requiring specific tokenizer libraries, making them suitable for
general-purpose use across different language models.

Key Features:
    - Word-based tokenization with subword estimation
    - Punctuation and special character handling
    - Language-aware token counting
    - Configurable estimation parameters
    - Fallback mechanisms for edge cases

The algorithms are based on empirical analysis of common tokenization patterns
used by modern language models like GPT, Claude, and others.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class TokenEstimationConfig:
    """Configuration for token estimation algorithms."""
    
    # Base estimation parameters
    avg_chars_per_token: float = 3.5  # More accurate than 4.0 for most models
    punctuation_token_ratio: float = 0.8  # Punctuation often tokenizes to fewer tokens
    number_token_ratio: float = 1.2  # Numbers often tokenize to more tokens
    
    # Subword estimation
    enable_subword_estimation: bool = True
    long_word_threshold: int = 8  # Words longer than this likely split into subwords
    subword_split_ratio: float = 1.4  # Multiplier for long words
    
    # Special patterns
    code_token_multiplier: float = 1.3  # Code tends to have more tokens per character
    url_token_multiplier: float = 0.9  # URLs often tokenize efficiently
    
    # Language-specific adjustments
    enable_language_detection: bool = True
    non_english_multiplier: float = 1.1  # Non-English text often needs more tokens


class TokenEstimator:
    """
    Advanced token estimation using word-based analysis and pattern recognition.
    
    This estimator provides more accurate token counts than simple character division
    by analyzing text patterns, word structure, and content type.
    """
    
    def __init__(self, config: Optional[TokenEstimationConfig] = None):
        """
        Initialize the token estimator.
        
        :param config: Configuration for estimation parameters
        :type config: Optional[TokenEstimationConfig]
        """
        self.config = config or TokenEstimationConfig()
        
        # Compile regex patterns for efficiency
        self._word_pattern = re.compile(r'\b\w+\b')
        self._punctuation_pattern = re.compile(r'[^\w\s]')
        self._number_pattern = re.compile(r'\b\d+\b')
        self._code_pattern = re.compile(r'```|`[^`]+`|def |class |import |function\(')
        self._url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        self._whitespace_pattern = re.compile(r'\s+')
        
        # Common programming keywords and patterns that affect tokenization
        self._code_keywords = {
            'def', 'class', 'import', 'from', 'function', 'const', 'let', 'var',
            'public', 'private', 'static', 'async', 'await', 'return', 'if', 'else'
        }
    
    def estimate_tokens(self, text: str, context: str = "general") -> int:
        """
        Estimate the number of tokens in the given text.
        
        :param text: The text to analyze
        :type text: str
        :param context: Context hint for better estimation ("general", "code", "prompt", "response")
        :type context: str
        :return: Estimated number of tokens
        :rtype: int
        """
        if not text or not text.strip():
            return 0
        
        # Get base word count and analyze patterns
        analysis = self._analyze_text(text)
        
        # Calculate base token estimate
        base_tokens = self._calculate_base_tokens(analysis)
        
        # Apply context-specific adjustments
        adjusted_tokens = self._apply_context_adjustments(base_tokens, analysis, context)
        
        # Apply final adjustments and ensure minimum
        final_tokens = max(1, int(round(adjusted_tokens)))
        
        logger.debug(f"Token estimation: '{text[:50]}...' -> {final_tokens} tokens")
        return final_tokens
    
    def estimate_prompt_tokens(self, prompt: str, system_prompt: Optional[str] = None) -> int:
        """
        Estimate tokens for a complete prompt including system prompt.
        
        :param prompt: The user prompt
        :type prompt: str
        :param system_prompt: Optional system prompt
        :type system_prompt: Optional[str]
        :return: Estimated total prompt tokens
        :rtype: int
        """
        total_tokens = self.estimate_tokens(prompt, context="prompt")
        
        if system_prompt:
            total_tokens += self.estimate_tokens(system_prompt, context="prompt")
            # Add small overhead for prompt formatting
            total_tokens += 5
        
        return total_tokens
    
    def _analyze_text(self, text: str) -> Dict:
        """
        Analyze text patterns for better token estimation.
        
        :param text: Text to analyze
        :type text: str
        :return: Analysis results
        :rtype: Dict
        """
        # Basic counts
        words = self._word_pattern.findall(text)
        punctuation_count = len(self._punctuation_pattern.findall(text))
        number_count = len(self._number_pattern.findall(text))
        
        # Pattern detection
        has_code = bool(self._code_pattern.search(text))
        has_urls = bool(self._url_pattern.search(text))
        
        # Word analysis
        long_words = [w for w in words if len(w) > self.config.long_word_threshold]
        code_words = [w for w in words if w.lower() in self._code_keywords]
        
        # Character analysis
        char_count = len(text)
        whitespace_count = len(self._whitespace_pattern.findall(text))
        
        return {
            'word_count': len(words),
            'char_count': char_count,
            'punctuation_count': punctuation_count,
            'number_count': number_count,
            'long_words': len(long_words),
            'code_words': len(code_words),
            'has_code': has_code,
            'has_urls': has_urls,
            'whitespace_count': whitespace_count,
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
        }
    
    def _calculate_base_tokens(self, analysis: Dict) -> float:
        """
        Calculate base token estimate from text analysis.
        
        :param analysis: Text analysis results
        :type analysis: Dict
        :return: Base token estimate
        :rtype: float
        """
        # Start with word-based estimation
        base_tokens = analysis['word_count']
        
        # Add punctuation tokens (usually fewer than character count)
        base_tokens += analysis['punctuation_count'] * self.config.punctuation_token_ratio
        
        # Adjust for numbers (often tokenize differently)
        number_adjustment = analysis['number_count'] * (self.config.number_token_ratio - 1)
        base_tokens += number_adjustment
        
        # Adjust for long words (likely to be split into subwords)
        if self.config.enable_subword_estimation and analysis['long_words'] > 0:
            subword_adjustment = analysis['long_words'] * (self.config.subword_split_ratio - 1)
            base_tokens += subword_adjustment
        
        # Fallback to character-based estimation if word count is very low
        if analysis['word_count'] < 3:
            char_based = analysis['char_count'] / self.config.avg_chars_per_token
            base_tokens = max(base_tokens, char_based)
        
        return base_tokens
    
    def _apply_context_adjustments(self, base_tokens: float, analysis: Dict, context: str) -> float:
        """
        Apply context-specific adjustments to token estimate.
        
        :param base_tokens: Base token estimate
        :type base_tokens: float
        :param analysis: Text analysis results
        :type analysis: Dict
        :param context: Context hint
        :type context: str
        :return: Adjusted token estimate
        :rtype: float
        """
        adjusted_tokens = base_tokens
        
        # Code context adjustments
        if context == "code" or analysis['has_code']:
            adjusted_tokens *= self.config.code_token_multiplier
        
        # URL adjustments
        if analysis['has_urls']:
            adjusted_tokens *= self.config.url_token_multiplier
        
        # Prompt context (often has special formatting)
        if context == "prompt":
            # Prompts often have more structured text
            adjusted_tokens *= 1.05
        
        # Response context (often more natural language)
        elif context == "response":
            # Responses tend to be more efficient in tokenization
            adjusted_tokens *= 0.95
        
        # Language detection adjustment (simplified)
        if self.config.enable_language_detection:
            # Simple heuristic: if average word length is high, might be non-English
            if analysis['avg_word_length'] > 6:
                adjusted_tokens *= self.config.non_english_multiplier
        
        return adjusted_tokens


# Global instance for easy access
default_estimator = TokenEstimator()


def estimate_tokens(text: str, context: str = "general") -> int:
    """
    Convenience function for token estimation using default configuration.
    
    :param text: Text to estimate tokens for
    :type text: str
    :param context: Context hint for better estimation
    :type context: str
    :return: Estimated number of tokens
    :rtype: int
    """
    return default_estimator.estimate_tokens(text, context)


def estimate_prompt_tokens(prompt: str, system_prompt: Optional[str] = None) -> int:
    """
    Convenience function for prompt token estimation.
    
    :param prompt: The user prompt
    :type prompt: str
    :param system_prompt: Optional system prompt
    :type system_prompt: Optional[str]
    :return: Estimated total prompt tokens
    :rtype: int
    """
    return default_estimator.estimate_prompt_tokens(prompt, system_prompt)

"""
Claude API-specific caching with semantic similarity detection.

Provides intelligent caching for Claude API calls with the ability to
detect semantically similar prompts and reuse responses when appropriate.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import logging

from kosmos.core.cache import HybridCache
from kosmos.core.cache_manager import CacheType, get_cache_manager

logger = logging.getLogger(__name__)


class ClaudePromptNormalizer:
    """
    Normalize Claude prompts for better cache hit rates.

    Handles whitespace normalization, placeholder replacement,
    and other transformations to make similar prompts cacheable.
    """

    @staticmethod
    def normalize(prompt: str, aggressive: bool = False) -> str:
        """
        Normalize a Claude prompt.

        Args:
            prompt: Original prompt text
            aggressive: If True, perform more aggressive normalization
                       (may affect response quality)

        Returns:
            Normalized prompt
        """
        # Basic normalization
        normalized = prompt.strip()

        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Normalize line endings
        normalized = normalized.replace('\r\n', '\n')
        normalized = re.sub(r'\n\s*\n', '\n\n', normalized)

        if aggressive:
            # Remove inline comments (be careful, might affect code)
            normalized = re.sub(r'#.*?$', '', normalized, flags=re.MULTILINE)

            # Normalize case for certain keywords (risky)
            # This is disabled by default as it can change semantics
            pass

        return normalized

    @staticmethod
    def extract_template(prompt: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract template and variables from a prompt.

        Useful for detecting similar prompts with different values.

        Args:
            prompt: Original prompt

        Returns:
            Tuple of (template, variables dict)
        """
        variables = {}

        # Detect common patterns (numbers, dates, IDs, etc.)
        template = prompt

        # Replace numbers with placeholders
        number_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(number_pattern, template)
        for i, num in enumerate(numbers):
            var_name = f"__NUM_{i}__"
            variables[var_name] = num
            template = template.replace(num, var_name, 1)

        # Replace quoted strings
        string_pattern = r'"([^"]+)"'
        strings = re.findall(string_pattern, template)
        for i, s in enumerate(strings):
            var_name = f"__STR_{i}__"
            variables[var_name] = s
            template = re.sub(string_pattern, f'"{var_name}"', template, count=1)

        return template, variables

    @staticmethod
    def compute_similarity_simple(prompt1: str, prompt2: str) -> float:
        """
        Compute simple similarity between two prompts.

        Uses character-level similarity (not semantic).

        Args:
            prompt1: First prompt
            prompt2: Second prompt

        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both prompts
        norm1 = ClaudePromptNormalizer.normalize(prompt1)
        norm2 = ClaudePromptNormalizer.normalize(prompt2)

        # Compute character-level overlap
        set1 = set(norm1.lower().split())
        set2 = set(norm2.lower().split())

        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        # Jaccard similarity
        return len(intersection) / len(union)


class ClaudeCache:
    """
    Intelligent cache for Claude API responses.

    Features:
    - Content-based caching with normalization
    - Semantic similarity detection (simple version)
    - Separate caching for API and CLI modes
    - Response metadata tracking
    - Cache bypass for specific prompts
    """

    def __init__(
        self,
        enable_normalization: bool = True,
        enable_similarity: bool = False,  # Disabled by default (no embeddings)
        similarity_threshold: float = 0.95,
        cache_bypass_patterns: Optional[List[str]] = None
    ):
        """
        Initialize Claude cache.

        Args:
            enable_normalization: Enable prompt normalization
            enable_similarity: Enable similarity-based caching
            similarity_threshold: Minimum similarity for cache hit (0-1)
            cache_bypass_patterns: Regex patterns that bypass cache
        """
        self.enable_normalization = enable_normalization
        self.enable_similarity = enable_similarity
        self.similarity_threshold = similarity_threshold
        self.cache_bypass_patterns = cache_bypass_patterns or [
            r'current\s+(time|date)',  # Time-sensitive queries
            r'random|generate\s+random',  # Random generation
            r'latest|newest|most\s+recent',  # Latest information
        ]

        # Get the underlying cache from cache manager
        self.cache_manager = get_cache_manager()
        self.cache = self.cache_manager.get_cache(CacheType.CLAUDE)

        # Normalizer
        self.normalizer = ClaudePromptNormalizer()

        logger.info(
            f"ClaudeCache initialized: "
            f"normalization={enable_normalization}, "
            f"similarity={enable_similarity}"
        )

    def _should_bypass_cache(self, prompt: str) -> bool:
        """Check if prompt should bypass cache."""
        for pattern in self.cache_bypass_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.debug(f"Cache bypass: prompt matches pattern '{pattern}'")
                return True
        return False

    def _generate_cache_key(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> str:
        """
        Generate cache key for a Claude API call.

        Args:
            prompt: Prompt text
            model: Model name
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Cache key
        """
        # Normalize prompt if enabled
        if self.enable_normalization:
            prompt = self.normalizer.normalize(prompt)

        # Create key from prompt + model + params
        key_data = {
            'prompt': prompt,
            'model': model,
            'params': sorted(kwargs.items())
        }

        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        model: str,
        bypass: bool = False,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached Claude response.

        Args:
            prompt: Prompt text
            model: Model name
            bypass: Force bypass cache
            **kwargs: Additional Claude parameters

        Returns:
            Cached response dict with 'content', 'metadata', or None
        """
        # Check bypass
        if bypass or self._should_bypass_cache(prompt):
            logger.debug("Cache bypassed for prompt")
            return None

        # Generate cache key
        cache_key = self._generate_cache_key(prompt, model, **kwargs)

        # Try exact match first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Exact cache hit: {cache_key[:8]}...")
            cached['cache_hit_type'] = 'exact'
            return cached

        # Try similarity-based matching if enabled
        if self.enable_similarity:
            similar_response = self._find_similar_cached(prompt, model, **kwargs)
            if similar_response is not None:
                logger.info("Similarity-based cache hit")
                similar_response['cache_hit_type'] = 'similar'
                return similar_response

        logger.debug("Cache miss")
        return None

    def set(
        self,
        prompt: str,
        model: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Store a Claude response in cache.

        Args:
            prompt: Prompt text
            model: Model name
            response: Claude response text
            metadata: Optional metadata (tokens, latency, etc.)
            **kwargs: Additional Claude parameters

        Returns:
            True if cached successfully
        """
        # Don't cache if should bypass
        if self._should_bypass_cache(prompt):
            return False

        # Generate cache key
        cache_key = self._generate_cache_key(prompt, model, **kwargs)

        # Prepare cache data
        cache_data = {
            'prompt': prompt,
            'model': model,
            'response': response,
            'metadata': metadata or {},
            'params': kwargs
        }

        # Store in cache
        success = self.cache.set(cache_key, cache_data)

        if success:
            logger.debug(f"Cached Claude response: {cache_key[:8]}...")

        return success

    def _find_similar_cached(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Find a similar cached response using simple similarity.

        This is a basic implementation that doesn't use embeddings.
        For production, consider using sentence transformers.

        Args:
            prompt: Prompt text
            model: Model name
            **kwargs: Additional parameters

        Returns:
            Similar cached response or None
        """
        # Note: This is inefficient as it checks all cached items
        # For production, maintain a separate similarity index

        # Disabled for now (requires iterating all cache entries)
        # In production, you'd maintain a separate index or use embeddings

        return None

    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Regex pattern to match prompts

        Returns:
            Number of entries invalidated
        """
        # This requires iterating the cache, which is expensive
        # For now, just log the request
        logger.warning(
            f"Pattern-based invalidation requested: {pattern}. "
            "This operation is not yet implemented efficiently."
        )
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get Claude cache statistics."""
        base_stats = self.cache.get_stats()

        # Add Claude-specific stats
        base_stats.update({
            'normalization_enabled': self.enable_normalization,
            'similarity_enabled': self.enable_similarity,
            'similarity_threshold': self.similarity_threshold,
        })

        return base_stats

    def clear(self) -> int:
        """Clear all Claude cache entries."""
        return self.cache.clear()


# Global Claude cache instance
_claude_cache: Optional[ClaudeCache] = None


def get_claude_cache(
    enable_normalization: bool = True,
    enable_similarity: bool = False
) -> ClaudeCache:
    """
    Get or create the global Claude cache instance.

    Args:
        enable_normalization: Enable prompt normalization
        enable_similarity: Enable similarity-based caching

    Returns:
        ClaudeCache instance
    """
    global _claude_cache

    if _claude_cache is None:
        _claude_cache = ClaudeCache(
            enable_normalization=enable_normalization,
            enable_similarity=enable_similarity
        )

    return _claude_cache


def reset_claude_cache():
    """Reset the global Claude cache (useful for testing)."""
    global _claude_cache
    _claude_cache = None

"""Semantic text compression package."""

from .semantic_compression import (
    SemanticCompressor,
    CompressionResult,
    SentimentResult,
    preprocess,
    clean_sentence,
)

__all__ = [
    "SemanticCompressor",
    "CompressionResult",
    "SentimentResult",
    "preprocess",
    "clean_sentence",
]

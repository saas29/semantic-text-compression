"""
semantic_compression.py

Core implementation of the semantic text compression pipeline described in
"Meaning Matters: A Lightweight Semantic Text Compression Framework Using
NLP and Sentiment Analysis".

Pipeline (see Fig. 1 / Algorithm 1 of the paper):
    1. Pre-processing        -> sentence tokenization + cleaning (NLTK)
    2. Sentence embeddings   -> MiniLM (sentence-transformers)
    3. Semantic similarity   -> pairwise cosine similarity
    4. Clustering            -> Agglomerative Clustering (average linkage,
                                 cosine distance, threshold = 0.6)
    5. Representative select -> sentence closest to each cluster centroid
    6. Sentiment analysis    -> DistilBERT (before vs. after compression)
    7. Metrics               -> Compression Ratio (CR), Redundancy
                                 Reduction (RR), Semantic Similarity (SS),
                                 Sentiment Consistency (SC)

Usage:
    from semantic_compression import SemanticCompressor

    compressor = SemanticCompressor()
    result = compressor.compress(text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# NLTK setup (Punkt sentence tokenizer)
# ---------------------------------------------------------------------------

def _ensure_nltk_punkt() -> None:
    """Download the NLTK Punkt tokenizer models the first time they are needed."""
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(
                f"tokenizers/{resource}" if resource == "punkt" else f"tokenizers/{resource}"
            )
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                # Older NLTK versions do not ship punkt_tab; punkt alone is enough.
                pass


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    label: str
    score: float


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    original_sentences: List[str]
    compressed_sentences: List[str]
    num_original_sentences: int
    num_clusters: int
    compression_ratio: float
    redundancy_reduction: float
    semantic_similarity: float
    sentiment_consistency: int
    original_sentiment: SentimentResult
    compressed_sentiment: SentimentResult

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "compressed_text": self.compressed_text,
            "num_original_sentences": self.num_original_sentences,
            "num_clusters": self.num_clusters,
            "compression_ratio": round(self.compression_ratio, 3),
            "redundancy_reduction": round(self.redundancy_reduction, 2),
            "semantic_similarity": round(self.semantic_similarity, 3),
            "sentiment_consistency": self.sentiment_consistency,
            "original_sentiment": {
                "label": self.original_sentiment.label,
                "score": round(self.original_sentiment.score, 4),
            },
            "compressed_sentiment": {
                "label": self.compressed_sentiment.label,
                "score": round(self.compressed_sentiment.score, 4),
            },
        }


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{1,}")


def clean_sentence(sentence: str) -> str:
    """Strip whitespace/noise from a single sentence while preserving casing.

    Casing is preserved (rather than lower-cased) so the compressed output
    stays human-readable and DistilBERT sentiment scoring stays accurate;
    only whitespace and repeated punctuation noise are normalised.
    """
    sentence = sentence.strip()
    sentence = _WHITESPACE_RE.sub(" ", sentence)
    sentence = _REPEATED_PUNCT_RE.sub(r"\1", sentence)
    return sentence


def preprocess(text: str) -> List[str]:
    """Tokenize raw input text into a list of cleaned sentences."""
    _ensure_nltk_punkt()
    from nltk.tokenize import sent_tokenize

    text = text.strip()
    if not text:
        return []

    raw_sentences = sent_tokenize(text)
    cleaned = [clean_sentence(s) for s in raw_sentences]
    return [s for s in cleaned if s]


# ---------------------------------------------------------------------------
# Semantic Compressor
# ---------------------------------------------------------------------------

class SemanticCompressor:
    """Lightweight semantic text compression + sentiment consistency pipeline."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        distance_threshold: float = 0.6,
        linkage: str = "average",
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.sentiment_model_name = sentiment_model_name
        self.distance_threshold = distance_threshold
        self.linkage = linkage

        self._embedder = None
        self._sentiment_pipeline = None

    # -- lazy model loading -------------------------------------------------

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def sentiment_pipeline(self):
        if self._sentiment_pipeline is None:
            from transformers import pipeline

            self._sentiment_pipeline = pipeline(
                "sentiment-analysis", model=self.sentiment_model_name
            )
        return self._sentiment_pipeline

    def warm_up(self) -> None:
        """Force both models to load. Call once at app startup to avoid
        paying the load cost on the first user request."""
        _ = self.embedder
        _ = self.sentiment_pipeline

    # -- pipeline steps -------------------------------------------------

    def embed(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences into 384-d MiniLM embeddings."""
        return np.asarray(
            self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
        )

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Agglomerative Hierarchical Clustering with average linkage and a
        cosine-distance threshold (Algorithm 1, step 4)."""
        from sklearn.cluster import AgglomerativeClustering

        n = len(embeddings)
        if n <= 1:
            return np.zeros(n, dtype=int)

        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage=self.linkage,
                distance_threshold=self.distance_threshold,
            )
        except TypeError:
            # scikit-learn < 1.2 used `affinity` instead of `metric`.
            model = AgglomerativeClustering(
                n_clusters=None,
                affinity="cosine",
                linkage=self.linkage,
                distance_threshold=self.distance_threshold,
            )
        return model.fit_predict(embeddings)

    def select_representatives(
        self, sentences: List[str], embeddings: np.ndarray, labels: np.ndarray
    ) -> List[int]:
        """For each cluster, pick the sentence whose embedding is closest to
        the cluster centroid (Algorithm 1, step 5). Returns original-text
        indices in reading order so the compressed text stays coherent."""
        from sklearn.metrics.pairwise import cosine_similarity

        representative_indices = []
        for cluster_id in np.unique(labels):
            member_idx = np.where(labels == cluster_id)[0]
            if len(member_idx) == 1:
                representative_indices.append(int(member_idx[0]))
                continue

            cluster_embeddings = embeddings[member_idx]
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(cluster_embeddings, centroid).ravel()
            best_local_idx = member_idx[int(np.argmax(sims))]
            representative_indices.append(int(best_local_idx))

        return sorted(representative_indices)

    def analyze_sentiment(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult(label="NEUTRAL", score=0.0)
        result = self.sentiment_pipeline(text, truncation=True)[0]
        return SentimentResult(label=result["label"], score=float(result["score"]))

    # -- metrics -------------------------------------------------

    @staticmethod
    def _semantic_similarity(original_embeddings: np.ndarray, compressed_embeddings: np.ndarray) -> float:
        from sklearn.metrics.pairwise import cosine_similarity

        mean_original = original_embeddings.mean(axis=0, keepdims=True)
        mean_compressed = compressed_embeddings.mean(axis=0, keepdims=True)
        return float(cosine_similarity(mean_original, mean_compressed)[0][0])

    # -- top-level entry point -------------------------------------------------

    def compress(self, text: str) -> CompressionResult:
        original_sentences = preprocess(text)
        n = len(original_sentences)

        if n == 0:
            empty_sentiment = SentimentResult(label="NEUTRAL", score=0.0)
            return CompressionResult(
                original_text=text,
                compressed_text="",
                original_sentences=[],
                compressed_sentences=[],
                num_original_sentences=0,
                num_clusters=0,
                compression_ratio=0.0,
                redundancy_reduction=0.0,
                semantic_similarity=0.0,
                sentiment_consistency=1,
                original_sentiment=empty_sentiment,
                compressed_sentiment=empty_sentiment,
            )

        embeddings = self.embed(original_sentences)
        labels = self.cluster(embeddings)
        representative_indices = self.select_representatives(original_sentences, embeddings, labels)

        compressed_sentences = [original_sentences[i] for i in representative_indices]
        compressed_embeddings = embeddings[representative_indices]

        compressed_text = " ".join(compressed_sentences)
        original_text_joined = " ".join(original_sentences)

        k = len(representative_indices)
        compression_ratio = (len(compressed_text) / len(original_text_joined)) if original_text_joined else 0.0
        redundancy_reduction = (1 - (k / n)) * 100
        semantic_similarity = self._semantic_similarity(embeddings, compressed_embeddings)

        original_sentiment = self.analyze_sentiment(original_text_joined)
        compressed_sentiment = self.analyze_sentiment(compressed_text)
        sentiment_consistency = int(original_sentiment.label == compressed_sentiment.label)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_sentences=original_sentences,
            compressed_sentences=compressed_sentences,
            num_original_sentences=n,
            num_clusters=k,
            compression_ratio=compression_ratio,
            redundancy_reduction=redundancy_reduction,
            semantic_similarity=semantic_similarity,
            sentiment_consistency=sentiment_consistency,
            original_sentiment=original_sentiment,
            compressed_sentiment=compressed_sentiment,
        )


if __name__ == "__main__":
    sample_text = (
        "I went into this film with pretty low expectations, but I have to admit I was "
        "pleasantly surprised. The story, while not completely original, was engaging "
        "enough to keep me interested throughout. The lead actor did a fantastic job "
        "portraying a conflicted character, and the supporting cast really added depth "
        "to the narrative. Some scenes felt unnecessarily long and repetitive, but the "
        "emotional payoff in the final act was worth it. The cinematography was "
        "beautiful, and the soundtrack complemented the tone of the film perfectly. "
        "Overall, it's a film with flaws but one that leaves a lasting impression."
    )

    compressor = SemanticCompressor()
    result = compressor.compress(sample_text)

    print("Compressed text:\n", result.compressed_text)
    print("\nMetrics:", result.to_dict())

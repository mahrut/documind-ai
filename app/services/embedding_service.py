
"""Embedding service for DocuMind AI."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingService:
    """Generate normalized text embeddings."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        # Load the pretrained model once.
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _validate_text(text: str):
        if not isinstance(text, str):
            raise TypeError("Each input must be a string")

        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate one 384-dimensional embedding."""

        self._validate_text(text)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Expected shape ({EMBEDDING_DIMENSION},), "
                f"but received {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains invalid numerical values")

        return embedding

    def generate_embeddings(
        self, texts: Sequence[str]
    ) -> np.ndarray:
        """Generate embeddings for multiple texts."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a list or tuple")

        if len(texts) == 0:
            raise ValueError("Text list cannot be empty")

        for text in texts:
            self._validate_text(text)

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (len(texts), EMBEDDING_DIMENSION)

        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, "
                f"but received {embeddings.shape}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid numerical values")

        return embeddings

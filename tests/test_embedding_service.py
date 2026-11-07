
"""Tests for app.services.embedding_service."""

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    """Load the pretrained model once for all tests."""
    return EmbeddingService()


def test_generate_single_embedding(embedding_service):
    text = "Artificial intelligence improves productivity."

    embedding = embedding_service.generate_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_generate_multiple_embeddings(embedding_service):
    texts = [
        "Artificial intelligence improves productivity.",
        "Machine learning is a branch of AI.",
        "Python is a programming language.",
    ]

    embeddings = embedding_service.generate_embeddings(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_text_raises_value_error(embedding_service):
    with pytest.raises(ValueError, match="empty"):
        embedding_service.generate_embedding("")


def test_whitespace_text_raises_value_error(embedding_service):
    with pytest.raises(ValueError, match="empty"):
        embedding_service.generate_embedding("   ")


def test_non_string_text_raises_type_error(embedding_service):
    with pytest.raises(TypeError, match="string"):
        embedding_service.generate_embedding(123)


def test_empty_batch_raises_value_error(embedding_service):
    with pytest.raises(ValueError, match="empty"):
        embedding_service.generate_embeddings([])


def test_string_is_not_accepted_as_a_batch(embedding_service):
    with pytest.raises(TypeError, match="sequence"):
        embedding_service.generate_embeddings("This is one string")


def test_batch_with_empty_item_raises_value_error(embedding_service):
    with pytest.raises(ValueError, match="empty"):
        embedding_service.generate_embeddings(["valid text", "   "])


def test_batch_with_non_string_item_raises_type_error(embedding_service):
    with pytest.raises(TypeError, match="string"):
        embedding_service.generate_embeddings(["valid text", 123])


def test_related_text_scores_above_unrelated_text(embedding_service):
    texts = [
        "Machine learning helps doctors diagnose diseases.",
        "Artificial intelligence supports medical diagnosis.",
        "The weather is sunny today.",
    ]

    query = "How can AI help doctors identify illnesses?"

    document_embeddings = embedding_service.generate_embeddings(texts)
    query_embedding = embedding_service.generate_embedding(query)

    # Normalized vectors: dot product equals cosine similarity.
    similarities = document_embeddings @ query_embedding

    assert similarities[0] > similarities[2]

import numpy as np
import pytest

from app.services.vector_store import VectorStore


def test_add_and_search_chunks():

    store = VectorStore(dimension=3)

    chunks = [
        {"page": 1, "text": "AI in healthcare"},
        {"page": 2, "text": "Database management"},
        {"page": 3, "text": "Machine learning"}
    ]

    embeddings = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0.9, 0.1, 0]
    ], dtype=np.float32)

    store.add_chunks(chunks, embeddings)

    query = np.array([1, 0, 0], dtype=np.float32)

    results = store.search(query, top_k=2)

    assert len(results) == 2

    assert results[0]["metadata"]["text"] == "AI in healthcare"

    assert results[0]["metadata"]["page"] == 1


def test_empty_store():

    store = VectorStore(dimension=3)

    results = store.search(
        np.array([1, 0, 0], dtype=np.float32)
    )

    assert results == []


def test_invalid_top_k():

    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.search(
            np.array([1, 0, 0], dtype=np.float32),
            top_k=0
        )


def test_mismatched_chunks_and_embeddings():

    store = VectorStore(dimension=3)

    chunks = [
        {"page": 1, "text": "Test"}
    ]

    embeddings = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add_chunks(chunks, embeddings)

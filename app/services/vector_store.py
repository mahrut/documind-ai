
import faiss
import numpy as np


class VectorStore:
    """
    FAISS-based vector store for storing and searching document embeddings.

    Uses inner product similarity. If embeddings are normalized,
    inner product is equivalent to cosine similarity.
    """

    def __init__(self, dimension=384):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError("Dimension must be a positive integer")

        self.dimension = dimension

        # Create FAISS index using inner product
        self.index = faiss.IndexFlatIP(self.dimension)

        # Store document chunks and metadata
        self.metadata = []

    def add_chunks(self, chunks, embeddings):
        """
        Add document chunks and their corresponding embeddings.

        Args:
            chunks: List of document chunks or metadata.
            embeddings: 2D NumPy array of embeddings.
        """

        if chunks is None or len(chunks) == 0:
            raise ValueError("Chunks cannot be empty")

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Validate embedding dimensions
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Incorrect embedding dimension")

        # Validate chunk and embedding counts
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "Chunks and embeddings count must match"
            )

        # Validate embedding values
        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain invalid values"
            )

        # Ensure contiguous float32 array for FAISS
        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32
        )

        # Add embeddings to FAISS index
        self.index.add(embeddings)

        # Preserve corresponding document metadata
        self.metadata.extend(chunks)

    def search(self, query_embedding, top_k=3):
        """
        Search for the most similar document chunks.

        Args:
            query_embedding: Query embedding as a 1D array.
            top_k: Maximum number of results to return.

        Returns:
            List of dictionaries containing similarity scores
            and corresponding metadata.
        """

        # Validate top_k
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer"
            )

        # Return empty results if the index is empty
        if self.index.ntotal == 0:
            return []

        # Convert query to float32
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        # Validate query dimensions
        if query_embedding.shape != (self.dimension,):
            raise ValueError(
                "Invalid query embedding shape"
            )

        # Validate query values
        if not np.isfinite(query_embedding).all():
            raise ValueError(
                "Query embedding contains invalid values"
            )

        # FAISS expects a 2D contiguous array
        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1),
            dtype=np.float32
        )

        # Determine the number of results to retrieve
        k = min(top_k, self.index.ntotal)

        # Search FAISS index
        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        # Collect search results
        for score, index in zip(scores[0], indices[0]):

            # Ignore invalid FAISS indices
            if index < 0:
                continue

            results.append({
                "score": float(score),
                "metadata": self.metadata[index]
            })

        return results

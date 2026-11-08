
from functools import lru_cache

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


@lru_cache(maxsize=1)
def get_services():
    """
    Initialize and return shared application services.

    The same service instances are reused throughout
    the application.
    """

    # 1. Initialize embedding service
    embedding_service = EmbeddingService()

    # 2. Initialize FAISS vector store
    vector_store = VectorStore(dimension=384)

    # 3. Initialize language model service
    llm_service = LLMService()

    # 4. Connect services to create RAG service
    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service
    )

    # 5. Return shared services
    return {
        "embedding": embedding_service,
        "vector_store": vector_store,
        "llm": llm_service,
        "rag": rag_service
    }

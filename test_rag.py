
"""Integration test for the DocuMind AI RAG pipeline."""

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


def main():
    print("Initializing DocuMind AI RAG pipeline...")

    # STEP 1: Initialize all services.
    embedding_service = EmbeddingService()

    vector_store = VectorStore(dimension=384)

    # Initialize Google Gemini (no Ollama required).
    llm_service = LLMService()

    # STEP 2: Prepare sample document chunks.
    chunks = [
        {
            "page": 1,
            "chunk_id": 0,
            "text": (
                "Artificial intelligence is increasingly "
                "being used in healthcare."
            )
        },
        {
            "page": 2,
            "chunk_id": 1,
            "text": (
                "Machine learning algorithms can analyze "
                "medical images and assist doctors "
                "in identifying diseases."
            )
        },
        {
            "page": 3,
            "chunk_id": 2,
            "text": (
                "Database systems are used to store "
                "and manage healthcare information."
            )
        }
    ]

    # STEP 3: Generate document embeddings.
    print("\nGenerating document embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_service.generate_embeddings(texts)

    print("Embedding shape:", embeddings.shape)

    # STEP 4: Add document chunks to FAISS.
    print("\nAdding document chunks to FAISS...")

    vector_store.add_chunks(chunks, embeddings)

    print("Total indexed chunks:", vector_store.index.ntotal)

    # STEP 5: Initialize the RAG service.
    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service
    )

    # STEP 6: Ask a question.
    question = "How does machine learning help doctors?"

    print("\nQuestion:", question)

    # STEP 7: Retrieve relevant chunks and generate an answer.
    print("\nGenerating answer using Google Gemini...")

    result = rag_service.answer_question(
        question=question,
        top_k=3
    )

    # STEP 8: Display the generated answer.
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(result["answer"])

    # STEP 9: Display retrieved sources.
    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print(f"\n{source['source']}")
        print(f"Page: {source['page']}")
        print(f"Similarity score: {source['score']:.4f}")
        print(f"Text: {source['text']}")

    print("\n" + "=" * 60)
    print("RAG PIPELINE EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""RAG pipeline service for DocuMind AI."""


class RAGService:
    """
    Connect embedding generation, FAISS document retrieval,
    context construction, and Gemini answer generation.
    """

    def __init__(
        self,
        embedding_service,
        vector_store,
        llm_service
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service

    def answer_question(self, question, top_k=3):
        """
        Answer a question using relevant chunks retrieved
        from the uploaded documents.
        """

        # STEP 1: Validate question
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()

        # STEP 2: Validate top_k
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        # STEP 3: Generate question embedding
        query_embedding = self.embedding_service.generate_embedding(
            question
        )

        # STEP 4: Retrieve relevant chunks from FAISS
        retrieved_results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # STEP 5: Handle empty search results
        if not retrieved_results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                "sources": []
            }

        context_parts = []
        sources = []

        # STEP 6: Build context and collect source information
        for index, result in enumerate(retrieved_results, start=1):

            # Validate search result
            if not isinstance(result, dict):
                continue

            metadata = result.get("metadata")

            if not isinstance(metadata, dict):
                continue

            # Extract document information
            page_number = metadata.get("page")
            chunk_text = metadata.get("text")

            filename = metadata.get(
                "filename",
                "Unknown document"
            )

            document_id = metadata.get("document_id")

            # Skip invalid or empty chunks
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                continue

            chunk_text = chunk_text.strip()

            source_label = f"Source {index}"

            # Add retrieved text to context
            context_parts.append(
                f"[{source_label} | {filename} | Page {page_number}]\n"
                f"{chunk_text}"
            )

            # Collect source metadata
            sources.append({
                "source": source_label,
                "filename": filename,
                "document_id": document_id,
                "page": page_number,
                "text": chunk_text,
                "score": result.get("score")
            })

        # STEP 7: Handle invalid or empty context
        if not context_parts:
            return {
                "answer": (
                    "I could not find usable document text "
                    "to answer this question."
                ),
                "sources": []
            }

        # STEP 8: Combine retrieved chunks
        context = "\n\n".join(context_parts)

        # STEP 9: Generate answer using Gemini
        answer = self.llm_service.generate_answer(
            question=question,
            context=context
        )

        # STEP 10: Return answer and sources
        return {
            "answer": answer,
            "sources": sources
        }

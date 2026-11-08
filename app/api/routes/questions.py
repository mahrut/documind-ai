
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.container import get_services


router = APIRouter()


class QuestionRequest(BaseModel):
    """
    Request model for document-related questions.
    """

    question: str = Field(
        min_length=1,
        max_length=2000
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )


@router.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Answer a question using the RAG pipeline.
    """

    try:
        # Step 1: Get shared services
        services = get_services()

        # Step 2: Execute RAG pipeline
        result = services["rag"].answer_question(
            request.question.strip(),
            top_k=request.top_k
        )

        # Step 3: Validate RAG response
        if not isinstance(result, dict):
            raise RuntimeError(
                "RAG service returned an invalid response."
            )

        if "answer" not in result or "sources" not in result:
            raise RuntimeError(
                "RAG response must contain answer and sources."
            )

        # Step 4: Return response
        return {
            "question": request.question.strip(),
            "answer": result["answer"],
            "sources": result["sources"],
            "status": "success"
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {error}"
        ) from error

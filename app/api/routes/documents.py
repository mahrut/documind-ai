
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.container import get_services
from app.services.document_processor import extract_text_from_pdf
from app.services.text_processor import process_document_pages


router = APIRouter()

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its text, generate embeddings,
    and store the resulting chunks in FAISS.
    """

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    document_id = str(uuid4())
    original_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / f"{document_id}.pdf"

    total_size = 0

    try:
        # Step 1: Save the uploaded PDF
        with open(file_path, "wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF must not exceed 10 MB."
                    )

                destination.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty."
            )

        # Step 2: Extract text from PDF
        pages = extract_text_from_pdf(str(file_path))

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No pages found in the PDF."
            )

        # Step 3: Process and divide text into chunks
        processed_chunks = process_document_pages(pages)

        if not processed_chunks:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in PDF."
            )

        # Step 4: Add document metadata
        for chunk_data in processed_chunks:
            if not chunk_data.get("text"):
                raise ValueError("A processed chunk has no text.")

            chunk_data["document_id"] = document_id
            chunk_data["filename"] = original_filename

        # Step 5: Initialize shared services
        services = get_services()

        # Step 6: Generate embeddings
        texts = [
            chunk_data["text"]
            for chunk_data in processed_chunks
        ]

        embeddings = services["embedding"].generate_embeddings(texts)

        # Step 7: Store chunks and embeddings in FAISS
        services["vector_store"].add_chunks(
            processed_chunks,
            embeddings
        )

        # Step 8: Return upload information
        return {
            "message": "Document uploaded and processed successfully.",
            "document_id": document_id,
            "filename": original_filename,
            "pages": len(pages),
            "chunks": len(processed_chunks),
            "status": "success"
        }

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {error}"
        ) from error

    finally:
        await file.close()

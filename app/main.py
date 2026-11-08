# from fastapi import FastAPI
# app = FastAPI()
# title = "DocuMind AI",
# description = "An intelleigent reaserch document assistant",
# version = "1.0.0"


# @app.get("/")
# def home():
#     return {"message": "Welcome to DocuMind AI", "status": "Application is running"}

#     @app.get("/health")
#     def health_check():
#         return {"status": "healthy"}


from fastapi import FastAPI

from app.api.routes import documents, questions


app = FastAPI(
    title="DocuMind AI",
    description=(
        "An intelligent research document assistant "
        "powered by RAG, FAISS and Gemini."
    ),
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to DocuMind AI",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# Register document upload routes
app.include_router(
    documents.router,
    tags=["Documents"]
)

# Register question-answering routes
app.include_router(
    questions.router,
    tags=["Questions"]
)


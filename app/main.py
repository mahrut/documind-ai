from fastapi import FastAPI
app = FastAPI()
title = "DocuMind AI",
description = "An intelleigent reaserch document assistant",
version = "1.0.0"


@app.get("/")
def home():
    return {"message": "Welcome to DocuMind AI", "status": "Application is running"}

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

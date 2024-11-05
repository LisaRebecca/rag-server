from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import openai
import logging
import asyncio
import uvicorn
import os
from rag import RAG
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Initialize RAG application
rag_app = RAG()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


# Root endpoint
@app.get("/")
async def root():
    return JSONResponse(
        content={"message": "Welcome to the FastAPI server"}, status_code=200
    )


# Health status endpoint
@app.get("/health")
async def health_status():
    return JSONResponse(content={"status": "healthy"}, status_code=200)


# Define a request model for the input data
class PromptRequest(BaseModel):
    prompt: str


# Update endpoint to accept JSON data
@app.post("/generate-response")
async def generate_response(request: PromptRequest):
    try:
        # Access the prompt text with request.prompt
        response = await asyncio.to_thread(lambda: rag_app.generate(request.prompt))
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="127.0.0.1",
        port=8080,  # Using a different port to avoid conflict
        log_level="info",
        ssl_certfile="server.crt",  # Path to SSL certificate
        ssl_keyfile="server.key",  # Path to SSL key
    )

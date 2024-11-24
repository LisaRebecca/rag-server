from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from Auth import Authenticate_User, status, timedelta, ACCESS_TOKEN_EXPIRE_MINUTES, Create_Access_Token, User, Depends, Get_Current_User
from rag import RAG
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import openai
import logging
import asyncio
import uvicorn
import os
import psutil
import time

# Metrics Storage
request_count = 0
error_count = 0
response_times = []

REQUEST_COUNT = Counter('request_count', ' - Total # Of Requests')
CPU_USAGE = Gauge("cpu_usage", " - Current CPU Usage Percentage")
MEMORY_USAGE = Gauge("memory_usage", " - Current Memory Usage Percentage")
REQUEST_LATENCY = Histogram("http_request_latency_seconds", " - HTTP Request Latency In Seconds", ["method", "endpoint", "http_status"])

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

# Define a request model for the input data
class PromptRequest(BaseModel):
    prompt: str

class QueryRequest(BaseModel):
    query: str

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
   
# Prometheus Health Tracking middleware 
@app.middleware("http")
async def Track_Metrics(request: QueryRequest, call_next):
    global request_count, error_count, response_times

    REQUEST_COUNT.inc()

    start_time = time.time()
    request_count += 1
    try:
        response = await call_next(request)
        process_time = time.time() - start_time # Prometheus Metric
    except Exception as e:
        error_count += 1
        raise HTTPException(status_code = 500, detail = str(e))
    finally:
        end_time = time.time()
        response_times.append((end_time - start_time) * 1000) # In Ms

    REQUEST_LATENCY.labels(
        method = request.method,
        endpoint = request.url.path,
        http_status = response.status_code
    ).observe(process_time)
    
    return response

# Health Tracking endpoint
@app.get("/health")
async def Metrics():
    import psutil
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), media_type = "text/plain")


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
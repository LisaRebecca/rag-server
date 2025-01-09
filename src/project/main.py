from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from server import fastapi_router
from server.Auth import Authenticate_User, status, timedelta, ACCESS_TOKEN_EXPIRE_MINUTES, Create_Access_Token, User, Depends, Get_Current_User
from rag.rag import RAG
from server.fastapi_router import index
from helpers.smart_cache import SmartCache
from helpers.exception import CustomException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import openai
import logging
import uvicorn
import os
import psutil
import sys

# Prometheus Metrics
REQUEST_COUNT = Counter('request_count', 'Total # of Requests')
ERROR_COUNT = Counter('error_count', 'Total # of Errors')
CPU_USAGE = Gauge("cpu_usage", "Current CPU Usage Percentage")
MEMORY_USAGE = Gauge("memory_usage", "Current Memory Usage Percentage")
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP Request Latency in Seconds",
    ["method", "endpoint", "http_status"]
)

app = FastAPI(
    title = "RAG APP Integrated With FastAPI server, and Routing Queries through FAU's Endpoint",
    description = "A FastAPI server Integrating a Vanilla RAG application",
    version = "1.0.0"
)

# Include API routers
app.include_router(fastapi_router.router)

# Initialize RAG application
rag_app = RAG()
cache = SmartCache(index = index)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a request model for the input data
class PromptRequest(BaseModel):
    prompt: str

class TokenRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str

# Root endpoint
@app.get("/")
async def root():
    return JSONResponse(
        content={"message": "Welcome to the FastAPI server"}, status_code=200
    )

# Update endpoint to accept JSON data
# @app.post("/generate-response")
# async def generate_response(request: PromptRequest):
#     try:
#         # Access the prompt text with request.prompt
#         response = await asyncio.to_thread(lambda: rag_app.generate(request.prompt))
#         return JSONResponse(content={"response": response}, status_code=200)
#     except Exception as e:
#         logger.error(f"Error generating response: {e}")
#         raise HTTPException(status_code=500, detail="Error generating response")
   
# Prometheus Health Tracking middleware 
# @app.middleware("http")
# async def track_metrics(request: QueryRequest, call_next):
#     start_time = time.time()
#     REQUEST_COUNT.inc()  # Increment request count
#     try:
#         response = await call_next(str(request))
#         process_time = time.time() - start_time
#         REQUEST_LATENCY.labels(
#             method=request.method,
#             endpoint=request.url.path,
#             http_status=response.status_code
#         ).observe(process_time)
#     except Exception as e:
#         ERROR_COUNT.inc()  # Increment error count
#         raise CustomException(e, sys)
#     return response

# Health Tracking endpoint
@app.get("/health")
async def health_status():
    # Collect CPU and Memory Usage
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    return {
        "status": "OK",
        "cpu_usage": f"{cpu_usage}%",
        "memory_usage": f"{memory_usage}%",
    }

@app.get("/metrics")
async def Metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), media_type = "text/plain")

@app.get("/cache")
async def Cache_Contents():
    try:
        cache_contents = cache.get_cache_contents()
        if not cache_contents:
            return {"Message: Cache is Empty!"}
        return {"Cache Contents: ": cache_contents}
    except Exception as e:
        raise CustomException(e, sys)

@app.post("/token")
async def Login_For_Access_Token(request: TokenRequest):
    User = await Authenticate_User(request.username, request.password)
    if not User:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect username or password",
        )
    Access_Token_Expires = timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    Access_Token = await Create_Access_Token(
        data = {"sub": User.username}, expires_delta = Access_Token_Expires
    )
    return {"Access_Token": Access_Token, "token_type": "bearer"}
    
@app.get("/secure-data")
async def Secure_Data(current_user: User = Depends(Get_Current_User)):
    return {"message": f"Hello, {current_user.username}! This is Protected Data..."}


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

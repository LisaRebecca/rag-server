from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from server import fastapi_router
from server import embedding_service
from server.Auth import Authenticate_User, status, timedelta, ACCESS_TOKEN_EXPIRE_MINUTES, Create_Access_Token, User, Depends, Get_Current_User
from server.university_api import get_model_config
from server.fastapi_router import cache_index
from helpers.smart_cache import SmartCache
from helpers.exception import CustomException
from helpers.utils import load_all_model_configs
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import openai
import logging
import uvicorn
import os
import psutil
import sys
import json
from pydantic import BaseModel

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

# CORS Configuration
class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Extract origin from request headers
        origin = request.headers.get("origin", "")
        response = await call_next(request)
        
        # Check if the origin is allowed
        allowed_origins = ["http://localhost:8080", "http://localhost:8090"]
        if origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

# Add the custom middleware
app.add_middleware(CustomCORSMiddleware)

# Explicitly handle OPTIONS requests for preflight
@app.options("/v1/chat/completions")
async def preflight_handler(request: Request):
    origin = request.headers.get("origin", "*")
    headers = {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Allow-Credentials": "true",
    }
    logging.info(f"Handled OPTIONS request for /v1/chat/completions")
    logging.info(f"Response Headers: {headers}")
    return Response(content="Preflight OK", headers=headers)


# Include API routers
app.include_router(fastapi_router.router)
app.include_router(embedding_service.router)

# Initialize RAG application
cache = SmartCache(index = cache_index)

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

# Retrieve Models
@app.get("/v1/models") # "/v1/models" /v1/chat/completions/models
async def list_models():
    try:
        # Path to your config.json file
        CONFIG_PATH = "config.json"
        
        # Load all model configurations
        all_configs = load_all_model_configs(CONFIG_PATH)
        llm_configs = all_configs.get("llm", {})
        models_data = []
        for model_name, config in llm_configs.items():
            base_url = config.get("base_url")
            api_key = config.get("api_key")
            model_id = config.get("model")
            logging.info(f"Model '{model_name}' - API KEY: {api_key}, Base URL: {base_url}, Model ID: {model_id}")
            models_data.append({
                "id": model_id,
                "object": "model",
                "owned_by": "user",
                "name": model_name,  # Optionally include the model name
                "base_url": base_url  # Optionally include other details
            })
        
        return {
            "data": models_data
        }
    except FileNotFoundError as fnf_error:
        logging.error(f"Config file not found: {fnf_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration file not found.",
        )
    except json.JSONDecodeError as json_error:
        logging.error(f"Error decoding JSON: {json_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error decoding configuration file.",
        )
   
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
        port=8090,  # Using a different port to avoid conflict
        log_level="info",
        # ssl_certfile="server.crt",  # Path to SSL certificate
        # ssl_keyfile="server.key",  # Path to SSL key
    )
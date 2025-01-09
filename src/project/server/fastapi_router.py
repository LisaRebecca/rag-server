from fastapi import APIRouter
from pydantic import BaseModel

from rag.vanilla_RAG import generation, tokenizer, model
from rag.rag_retrieval import RAG_Retrieval
from helpers.utils import load_vector_db
from server.university_api import query_university_endpoint

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.smart_cache import SmartCache

import sys
import time
import os
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

# Authentication
API_KEYS = os.getenv('OPENAI_API_KEY')
if not API_KEYS:
    raise ValueError("API key is not set. Check the OPENAI_API_KEY environment variable.")

def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        logging.warning("No authorization header provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ")[1]
    if token not in API_KEYS:
        logging.warning(f"Invalid API key attempted: {token}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

VECTORSTORE_PATH = "vector_index_fau.faiss"
METADATA_PATH = "metadata.json" # Mock chunked data [text, metadata[source]]
METADATA_PATH_FAU = "knowledgebase/quality_html-pdf.jsonl" # FAU chunked data [text, url, file_path, chunk_no, dl_date, chunk_date, quality_score]

router = APIRouter()

index, metadata = load_vector_db(index_file = VECTORSTORE_PATH, metadata_file = METADATA_PATH_FAU)

cache = SmartCache(index = index)
                  
# Pydantic Models aligned with OpenAI's Completion API
class CompletionRequest(BaseModel):
    model: Optional[str] = Field(default="default-model", description="Model to use for completion")
    prompt: str = Field(..., description="The prompt to generate completions for")
    max_tokens: Optional[int] = Field(default=100, description="The maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling probability")
    n: Optional[int] = Field(default=1, description="Number of completions to generate")
    stop: Optional[List[str]] = Field(default=None, description="Up to 4 sequences where the API will stop generating further tokens")
    echo: Optional[bool] = Field(default=False, description="Whether to echo the prompt in the response")

class Choice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[dict] = None

@router.post("/v1/chat/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest, api_key: str = Depends(verify_api_key)):
    logging.info("Received completion request")
    try:
        start_time = time.time()

        logging.info(f"Model requested: {request.model}")
        logging.info(f"Prompt: {request.prompt}")

        # Step 1: Retrieval using RAG
        retrieved_docs = RAG_Retrieval.dense_retrieval(request.prompt, index, metadata, top_k=20)
        logging.info(f"Retrieved Documents: {retrieved_docs}")

        # Step 2: Generation using RAG
        rag_response = generation(request.prompt, retrieved_docs, tokenizer, model)
        logging.info(f"RAG Response: {rag_response}")

        # Step 3: Call University API
        rag_query = f"Based on the following documents {retrieved_docs}, please answer this question: {request.prompt}."
        uni_response = await query_university_endpoint(rag_query, 'techxgenus')
        logging.info(f"University Response: {uni_response}")

        end_time = time.time()
        logging.info(f"Finished query in: {end_time - start_time} seconds")

        # Structure the response similar to OpenAI's CompletionResponse
        response = CompletionResponse(
            id=str(uuid.uuid4()),  # Generate a unique ID as needed
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    text=f"University Response:\n{uni_response}\n",
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(uni_response.split()),
                "total_tokens": len(request.prompt.split()) + len(uni_response.split())
            }
        )
        output = (
            f"Query:\n{request.prompt}\n\n"
            # f"Our Answer:\n{rag_response}\n\n"
            f"University Response:\n{uni_response}\n"
        )
        print(output)
        cache.append_to_cache(request.prompt, uni_response)
        #logging.info("Structured response similar to OpenAI's CompletionResponse", response)
        
        return response 

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {ce}")
        raise ce  # Already handled by custom exception handler

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )

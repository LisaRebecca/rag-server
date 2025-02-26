from fastapi import APIRouter, Depends, HTTPException, status, Header
from pydantic import BaseModel, Field, Extra


from rag.vanilla_RAG import generation, tokenizer, model
from rag.rag_retrieval import RAG_Retrieval
from helpers.utils import load_vector_db, verify_api_key
from server.university_api import query_university_endpoint
from embedding_service.app.main import embedding_model_instance
from typing import Optional, List, Dict, Any

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.smart_cache import SmartCache

import numpy as np
import faiss
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
cache_index = faiss.IndexFlatL2(128)
cache = SmartCache(index = cache_index)
                  
# --------------------------------------------------------------
# 1) Define new models to match your incoming payload
# --------------------------------------------------------------
class Message(BaseModel):
    id: Optional[str] = None
    parentId: Optional[str] = None
    childrenIds: Optional[List[str]] = None
    role: Optional[str] = None
    content: Optional[str] = None
    model: Optional[str] = None
    modelName: Optional[str] = None
    modelIdx: Optional[int] = None
    userContext: Optional[Any] = None
    timestamp: Optional[int] = None
    done: Optional[bool] = None
    
    class Config:
        extra = Extra.allow  # ignore any extra fields in messages

class CompletionRequest(BaseModel):
    model: Optional[str] = Field(default="default-model")
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    type: Optional[str] = None
    stream: Optional[bool] = None

    # If you still want max_tokens, temperature, etc., keep them:
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0

    class Config:
        extra = Extra.allow  # allow any additional fields (params, background_tasks, etc.)

# --------------------------------------------------------------
# 2) Define your Chat Completion Response models
# --------------------------------------------------------------
class ChatMessageResponse(BaseModel):
    role: str
    content: str

class ChatChoiceResponse(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoiceResponse]
    usage: Optional[Dict[str, int]] = None

# --------------------------------------------------------------
# 3) Main endpoint: parse either request.prompt or from messages
# --------------------------------------------------------------
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_completion(
    request: CompletionRequest,
    api_key: str = Depends(verify_api_key)  # or remove if you don't want auth
):
    logging.info("Received completion request")
    logging.info(request.model)
    # 3A) Decide how to get user_prompt
    user_prompt = request.prompt if request.prompt else ""

    # If you prefer to get the last "user" message if prompt is empty
    if (not user_prompt) and request.messages:
        for msg in reversed(request.messages):
            if msg.role and msg.role.lower() == "user" and msg.content:
                user_prompt = msg.content
                break

    if not user_prompt:
        raise HTTPException(
            status_code=422,
            detail="No 'prompt' or user message content found in the request."
        )

    try:
        start_time = time.time()

        logging.info(f"Model requested: {request.model}")
        logging.info(f"User's prompt: {user_prompt}")
        if request.model == "TechxGenus_Mistral-Large-Instruct-2407-AWQ":          
            logging.info("Response with RAG")

            # Step 1: RAG retrieval
            print(f"User Prompt: {user_prompt}")

            query_embedding = embedding_model_instance.get_embedding(user_prompt)

            query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

            retrieved_docs = await RAG_Retrieval.dense_retrieval(query_embedding, index, metadata, top_k=20)
            logging.info(f"Retrieved Documents: {retrieved_docs}")

            # Step 2: RAG generation - Ours
            rag_response = generation(user_prompt, retrieved_docs, tokenizer, model)
            logging.info(f"RAG Response: {rag_response}")

            # Step 3: Call University API (or skip if you want)
            rag_query = f"Based on the following documents {retrieved_docs}, please answer this question: {user_prompt}."
            uni_response = await query_university_endpoint(rag_query, 'FAU LLM 2.0')
            logging.info(f"{uni_response}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Response took {elapsed_time:.4f} seconds")
            logging.info(f"Finished query in: {end_time - start_time} seconds")
        else:
            logging.info("Response without RAG")
            uni_response = await query_university_endpoint(user_prompt, 'FAU LLM 2.0')
        # Prepare the response in Chat Completion format
        response = ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=request.model or "default-model",
            choices=[
                ChatChoiceResponse(
                    index=0,
                    message=ChatMessageResponse(
                        role="assistant",
                        content=f"{uni_response}\n"
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(user_prompt.split()),
                "completion_tokens": len(uni_response.split()),
                "total_tokens": len(user_prompt.split()) + len(uni_response.split())
            }
        )
        
        # Optionally cache
        cache.append_to_cache(user_prompt, uni_response)
        print(f"Query:\n{user_prompt}\n\nUniversity Response:\n{uni_response}\n")

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

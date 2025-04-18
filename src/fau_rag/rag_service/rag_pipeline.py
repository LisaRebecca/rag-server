from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, Extra

from retrievers.dense_retrieval import dense_retrieval_instance
from embedding_service.app.main import embedding_model_instance
from helpers.utils import load_vector_db
from rag_service.university_api import query_university_endpoint
from typing import Optional, List, Dict, Any

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.smart_cache import SmartCache
from urllib.parse import urlparse

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
print(f"Loaded API Key: {API_KEYS}")

if not API_KEYS:
    raise ValueError("API key is not set. Check the OPENAI_API_KEY environment variable.")
    
VECTORSTORE_PATH = "knowledgebase/vector_index_fau.faiss"
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
    request: CompletionRequest):
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
            query_embedding = embedding_model_instance.get_embedding(user_prompt)

            query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
            
            retrieved_docs = dense_retrieval_instance.dense_retrieval_reranking(query_embedding, index, metadata, top_k=20)
            logging.info(f"Retrieved Documents: {retrieved_docs}")

            urls = []            
            for doc in retrieved_docs:
                url = doc.get("url")
                logging.info(f"URL: {url}")
                if url:                  
                    if not url.startswith("http"):
                        url = "https://" + url.replace("\\", "/")
                    else: url = "https://" + url
                    # Remove trailing "index.html" if present
                    if url.endswith("index.html"):
                        url = url[:-10]  # Remove the last 10 characters ("index.html")
                        if not url.endswith("/"):
                            url += "/"  
                    parsed = urlparse(url)
                    if "fau.de" not in parsed.netloc:
                        logging.info(f"Ignoring URL (not from fau.de): {url}")
                        continue 
                    if url not in urls:  
                        urls.append(url)

            if urls:
                citation_markdown = "\n".join([f"- [{link}]({link})" for link in urls])
                citation_section = f"\n\n**Sources**\n{citation_markdown}\n"
            else:
                citation_section = ""
                
            info = []            
            for doc in retrieved_docs:
                text = doc.get("text")
                info.append(text)

            # Step 2: Call University API (or skip if you want)
            rag_query = (
                f"Based on the following documents {info}, please answer the question: {user_prompt}. "
                f"The answer should be written fully in English, regardless of the languages used in the documents. "
                f"Only use foreign words when necessary, such as names or official titles, but keep the overall narrative in English. "
                f"If the question is asked in another language, reply in that language instead."
            )
            uni_response = await query_university_endpoint(rag_query, 'FAU LLM 2.0')
            logging.info(f"{uni_response}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Response took {elapsed_time:.4f} seconds")
            print(f"Citations: {citation_section}")
            logging.info(f"Finished query in: {end_time - start_time} seconds")
        else:
            logging.info("Response without RAG")
            uni_response = await query_university_endpoint(user_prompt, 'FAU LLM 2.0')
            citation_section = ""

        # Append the citations to the final response content
        final_content = f"{uni_response}{citation_section}"

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
                        content=f"{uni_response}{citation_section}"
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
        print(f"Query:\n{user_prompt}\n\nUniversity Response with Citations:\n{final_content}\n")
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
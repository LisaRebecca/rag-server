from fastapi import (APIRouter,
                     HTTPException,
                     status)

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.smart_cache import SmartCache
from helpers.utils import load_vector_db
from helpers.schemas import(CompletionRequest,
                            ChatCompletionResponse,
                            ChatChoiceResponse,
                            ChatMessageResponse)

from retrievers.dense_retrieval import dense_retrieval_instance
from embedding_service.app.main import embedding_model_instance
from rag_service.university_api import query_university_endpoint
from constants import (VECTORSTORE_PATH,
                       METADATA_PATH_FAU,
                       API_KEYS)

import numpy as np
import faiss
import time
import uuid
from urllib.parse import urlparse

# Authentication
if not API_KEYS:
    raise ValueError("API key is not set. Check the OPENAI_API_KEY environment variable.")

# FastAPI Router
router = APIRouter()

# Loading Knowledgebase
index, metadata = load_vector_db(index_file = VECTORSTORE_PATH, metadata_file = METADATA_PATH_FAU)

# Cache Setup
cache_index = faiss.IndexFlatL2(128)
cache = SmartCache(index = cache_index)
                  

# --------------------------------------------------------------
# RAG Pipeline
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
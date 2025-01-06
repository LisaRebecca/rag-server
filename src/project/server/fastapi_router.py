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

VECTORSTORE_PATH = "vector_index_fau.faiss"
METADATA_PATH = "metadata.json" # Mock chunked data [text, metadata[source]]
METADATA_PATH_FAU = "knowledgebase/quality_html-pdf.jsonl" # FAU chunked data [text, url, file_path, chunk_no, dl_date, chunk_date, quality_score]

router = APIRouter()

index, metadata = load_vector_db(index_file = VECTORSTORE_PATH, metadata_file = METADATA_PATH_FAU)

cache = SmartCache(index = index)

# Pydantic Models for request and response validation:
class QueryRequest(BaseModel):
    query: str

# Endpoints
@router.post("/query")
async def handle_query(request: QueryRequest):
    try:
        start_time = time.time()
        # Check Cache
        cached_result = cache.retrieve_from_cache(request.query)
        if cached_result:
            end_time = time.time()
            logging.info("Answer retrieved from Cache!")
            print(f"Cache hit! Finished query in: {end_time - start_time}")
            return {
                "from_cache": True,
                "result": cached_result,
            }
        
        print(f"Loaded index type: {type(index)}")
        
        # Step 1: Retrieval
        retrieved_docs = RAG_Retrieval.dense_retrieval(request.query, index, metadata, top_k = 20)

        # Step 2: Generation
        rag_response = generation(request.query, retrieved_docs, tokenizer, model)

        rag_query = f"Based on the following documents {retrieved_docs}, please asnwer this question: {request.query}."

        # Step 3: Call University API
        uni_response = await query_university_endpoint(rag_query)
        end_time = time.time()
        print(f"Finished query in: {end_time - start_time}")

        output = (
            f"Query:\n{request.query}\n\n"
            # f"Our Answer:\n{rag_response}\n\n"
            f"University Response:\n{uni_response}\n"
        )
        print(output)

        cache.append_to_cache(request.query, uni_response)
        return {
            "from_cache": False,
            "result": output,
        }
    
    except Exception as e:
        raise CustomException(e, sys)
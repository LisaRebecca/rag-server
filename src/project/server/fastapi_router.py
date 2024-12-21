from fastapi import APIRouter, Request
from pydantic import BaseModel

from rag.vanilla_RAG import retrieval, generation, tokenizer, model
from helpers.utils import load_vector_db
from server.university_api import query_university_endpoint

from helpers.exception import CustomException
from helpers.logger import logging

import sys

VECTORSTORE_PATH = "vector_index_fau.faiss"
METADATA_PATH = "metadata.json" # Mock chunked data [text, metadata[source]]
METADATA_PATH_FAU = "knowledgebase/quality_html-pdf.jsonl" # FAU chunked data [text, url, file_path, chunk_no, dl_date, chunk_date, quality_score]

router = APIRouter()

index, metadata = load_vector_db(index_file = VECTORSTORE_PATH, metadata_file = METADATA_PATH_FAU)


# Pydantic Models for request and response validation:
class QueryRequest(BaseModel):
    query: str

# Endpoints
@router.post("/query")
async def handle_query(request: QueryRequest):

    # Step 1: Retrieval
    try:
        retrieved_docs = retrieval(request.query, index, metadata)

        # Step 2: Generation
        rag_response = generation(request.query, retrieved_docs, tokenizer, model)

        rag_query = f"Based on the following documents {retrieved_docs}, please asnwer this question: {request.query}."

        # Step 3: Call University API
        # uni_response = query_university_endpoint(request.query)
        uni_response = query_university_endpoint(rag_query)

        output = (
            f"Query:\n{request.query}\n\n"
            # f"Our Answer:\n{rag_response}\n\n"
            f"University Response:\n{uni_response}\n"
        )
        print(output)
        return output
    except Exception as e:
        raise CustomException(e, sys)
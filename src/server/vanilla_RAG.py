import json
import faiss
import sys
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as Np
from helpers.exception import CustomException
from helpers.logger import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from helpers.utils import load_vector_db

VECTORSTORE_PATH = "vector_index.faiss"
METADATA_PATH = "metadata.json" # Mock chunked data [text, metadata[source]]
METADATA_PATH_FAU = "quality_html-pdf.jsonl" # FAU chunked data [text, url, file_path, chunk_no, dl_date, chunk_date, quality_score]

def __init__(self):
    pass

def retrieval(query, index, metadata, top_k = 5):
    try:
        retriever = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # Hugging Face model
        # retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = retriever.encode([query])

        _, indices = index.search(Np.array(query_embedding, dtype = "float32"), top_k)

        retrieved_docs = [metadata[i]["text"] for i in indices[0]]

        print("Retrieved Documents:", retrieved_docs)

        logging.info("Documents Retrieved Successfully!")
        return retrieved_docs

    except Exception as e:
        raise CustomException(e, sys)

def generation(query, retrieved_docs, tokenizer, model):
    try:
        logging.disable(logging.CRITICAL)
        logging.info("Creating Prompt...")
        context = " ".join(retrieved_docs)
        prompt = f"Given the following context, answer the question.\n\nContext: {context}"

        print("Generated Prompt:", prompt)

        inputs = tokenizer(prompt, return_tensors = "pt", truncation = True, max_length = 256)
        outputs = model.generate(inputs["input_ids"], max_length = 50, num_beams = 3)

        logging.disable(logging.NOTSET)
        logging.info("Output Generated Successfully!")
        return tokenizer.decode(outputs[0], skip_special_tokens = True)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Obj = RAG()
    RETRIEVER_MODEL = "all-mpnet-base-v2"
    GENERATOR_MODEL = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)

    index, metadata = load_vector_db(index_file = VECTORSTORE_PATH, metadata_file = METADATA_PATH)

    Query = "What are the prerequisites for the Data Science course?" # Example Query

    retrieve = retrieval(Query, index, metadata)
    Answer = generation(Query, retrieve, tokenizer, model)
    
    print(f"Query: {Query}")
    print(f"Answer: {Answer}")

import sys
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from helpers.exception import CustomException
from helpers.logger import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from helpers.utils import load_vector_db

from rag.rag_retrieval import RAG_Retrieval

VECTORSTORE_PATH = "vector_index_fau.faiss"
METADATA_PATH_FAU = "knowledgebase/quality_html-pdf.jsonl" # FAU chunked data [text, url, file_path, chunk_no, dl_date, chunk_date, quality_score]

# Loading the models once
RETRIEVER_MODEL = "all-mpnet-base-v2"
GENERATOR_MODEL = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)

def generation(query, retrieved_docs, tokenizer, model):
    try:
        logging.disable(logging.CRITICAL)
        logging.info("Creating Prompt...")
        # context = " ".join(retrieved_docs)
        prompt = f"Please answer this question: {query}, given the following documents: {retrieved_docs}"

        # print("Generated Prompt:", prompt)

        inputs = tokenizer(prompt, return_tensors = "pt", truncation = True, max_length = 256)
        outputs = model.generate(inputs["input_ids"], max_length = 100, num_beams = 3)

        logging.disable(logging.NOTSET)
        logging.info("Output Generated Successfully!")

        return tokenizer.decode(outputs[0], skip_special_tokens = True)
    except Exception as e:
        raise CustomException(e, sys)
    
def main():
    index, metadata = load_vector_db(index_file = VECTORSTORE_PATH, metadata_file = METADATA_PATH_FAU)
    Query = "What are the prerequisites for the Data Science course?" # Example Query
    # Query = "Tell me about the RRZE" # Example Query
    # Query = "How do I get into the FAU?" # Example Query
    top_k = 5
    alpha = 0.5

    hybrid_ret = RAG_Retrieval()
    hybrid_retrieval = hybrid_ret.hybrid_retrieval(query = Query, metadata = metadata, top_k = top_k, alpha = alpha)


    # retrieve_dense = RAG_Retrieval.dense_retrieval(Query, index, metadata, 5)
    # retrieve_sparse = RAG_Retrieval.sparse_retrieval(Query, metadata, 5)

    # hybrid_retrieval = RAG_Retrieval.hybrid_retrieval(query = Query, metadata = metadata, top_k = 20, alpha = 0.5)

    Answer = generation(Query, hybrid_retrieval, tokenizer, model)
    
    print(f"Query: {Query}")
    print(f"Answer: {Answer}")


if __name__ == "__main__":
    main()
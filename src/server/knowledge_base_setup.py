import os
import json
import sys
import faiss
import numpy as Np
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_vector_db


class KnowledgeBase:
    def __init__(self):
        pass

    def load_data(self, data_directory):
        try:
            documents = []
            logging.info("Reading Data From Directory...")
            for file_name in os.listdir(data_directory):
                file_path = os.path.join(data_directory, file_name)
                with open(file_path, 'r', encoding = 'utf-8') as file:
                    documents.append({"text": file.read(), "metadata": {"source": file_name}})

            logging.info("Documents Appended Into A List!")
            return documents

        except Exception as e:
            raise CustomException(e, sys)
    
    def chunking(self, documents):
        try:
            text_splitter = CharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
            chunks = []

            for doc in documents:
                doc_chunks = text_splitter.split_text(doc["text"])
                for chunk in doc_chunks:
                    chunks.append({"text": chunk, "metadata": doc["metadata"]})
            
            logging.info("Documents Broken Down Into Chunks!")
            return chunks

        except Exception as e:
            raise CustomException(e, sys)
    
    def embedding(self, chunks, embedding_model):
        try:
            model = SentenceTransformer(embedding_model)
            embeddings = model.encode([chunk["text"] for chunk in chunks], show_progress_bar = True)

            logging.info("Embeddings Generation Complete!")
            return embeddings

        except Exception as e:
            raise CustomException(e, sys)

        
if __name__ == "__main__":
    Obj = KnowledgeBase()
    DATA_DIR = "./notebook/mock_data/"
    EMBEDDING_MODEL = "all-mpnet-base-v2"

    documents = Obj.load_data(DATA_DIR)
    chunks = Obj.chunking(documents)

    embeddings = Obj.embedding(chunks, EMBEDDING_MODEL)

    save_vector_db(embeddings, chunks, index_file = "vector_index.faiss")

    logging.info("Knowledge Base Setup Complete!")
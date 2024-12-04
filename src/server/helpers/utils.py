import os
import sys
import json
import numpy as Np
import pandas as Pd
import faiss
from src.exception import CustomException
from src.logger import logging

def save_vector_db(embeddings, chunks, index_file):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(Np.array(embeddings))
        faiss.write_index(index, index_file)

        with open("metadata.json", "w") as f:
                json.dump(chunks, f)
        logging.info("Data Saved To Vector DB!")

    except Exception as e:
        raise CustomException(e, sys)
    

def load_vector_db(index_file, metadata_file):
    try:
        index = faiss.read_index(index_file)
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        logging.info("Data Loaded From Vector DB Successfully!")
        return index, metadata
    
    except Exception as e:
         raise CustomException(e, sys)
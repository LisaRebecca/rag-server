from rag.rag_retrieval import retriever
from helpers.exception import CustomException
from helpers.logger import logging
import faiss
from faiss import IndexFlatL2
import sys
import json
import os
import numpy as np
cache_file = r"D:\Shenime\Shenimet\Data_Science\Master_thesis\github_2\rag-server\src\project\smart_cache.json"
#cache_file = "smart_cache.json"

class SmartCache:
    def __init__(self, index: IndexFlatL2 = faiss.IndexFlatL2(128), model_name = retriever, similarity_threshold = 0.8, cache_file = CACHE_FILE):
        self.model = model_name
        self.similarity_threshold = similarity_threshold
        self.cache = {}
        self.index = index
        self.keys = []
        self.cache_file = cache_file
        self.load_cache()

    def save_cache(self):      

        try:
            # Convert numpy ndarrays to lists for JSON serialization
            serializable_cache = {}
            for query, data in self.cache.items():
                serializable_cache[query] = {
                    "embedding": data["embedding"].tolist(),  # Convert ndarray to list
                    "result": data["result"]
                }
            with open(self.cache_file, "w") as f:
                json.dump(serializable_cache, f)
            logging.info("Cache saved to file successfully.")
        except Exception as e:
            logging.error(f"Error saving cache to file: {e}")
            raise CustomException(e, sys) 
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    #self.cache = json.load(f)
                    serializable_cache = json.load(f)
                
                # Reconstruct the cache with numpy ndarrays
                for query, data in serializable_cache.items():
                    embedding = np.array(data["embedding"], dtype='float32')  # Convert list back to ndarray
                    self.cache[query] = {
                        "embedding": embedding,
                        "result": data["result"]
                    }
                    
                logging.info("Cache loaded from file successfully.")
            except Exception as e:
                logging.error(f"Error loading cache from file: {e}")
        else:
            logging.info(f"No cache file found at {self.cache_file}. Starting with an empty cache.")

    def get_cache_contents(self):
        if not self.cache:
            self.load_cache()

        if not self.cache:
            return {"Cache Contents": None}
            
        cache_contents = []
        for query, data in self.cache.items():
            cache_contents.append({"query": query, "result": data["result"]})
            
        return cache_contents

    def append_to_cache(self, query, result):

        try:
            # Encode the query to get its embedding
            embedding = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            self.keys.append(query)
            logging.info(self.keys)
            self.index.add(embedding)
            self.cache[query] = {"embedding": embedding[0], "result": result}
            self.save_cache()
            
            """embedding = embedding[0].astype('float32')  # Ensure the embedding is float32
            logging.info(embedding)
            self.keys.append(query)
            logging.info("Query and Embedding added to Cache Successfully!")

            #self.index.add(embedding.reshape(1, -1))  # Add embedding to FAISS index
            self.cache[query] = {"embedding": embedding, "result": result}
            self.index.add(embedding.reshape(1, -1))
            #self.index.add(embedding)
            #self.cache[query] = {"embedding": embedding[0], "result": result}
            
            try:
                self.save_cache()
            except Exception as e:
                logging.info("self.save_cache()...")"""
            logging.info("Query and Embedding added to Cache Successfully!")
            print(f"Cache After Append: {self.cache}")
        except Exception as e:
            logging.info("Cache Not Updated...")
            #raise CustomException(e, sys)

    def retrieve_from_cache(self, query):
        try:
            self.load_cache()
            embedding = self.model.encode([query], convert_to_tensor = False, normalize_embeddings = True)
            embedding = embedding[0].astype('float32')  # Ensure the embedding is float32
            if len(self.keys) == 0:
                logging.info("No keys in cache.")
                return None
            
            distances, indices = self.index.search(embedding, k = 1)
            if distances[0][0] <= (1 - self.similarity_threshold):
                closest_query = self.keys[indices[0][0]]
                return self.cache[closest_query]["result"]
            
            return None
        except Exception as e:
            raise CustomException(e, sys)
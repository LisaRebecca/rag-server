from rag.rag_retrieval import retriever
from helpers.exception import CustomException
from helpers.logger import logging
import faiss
from faiss import IndexFlatL2
import sys
import json
import os

CACHE_FILE = "smart_cache.json"

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
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
            logging.info("Cache saved to file successfully.")
        except Exception as e:
            logging.error(f"Error saving cache to file: {e}")
            raise CustomException(e, sys) 
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
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
            embedding = self.model.encode([query], convert_to_tensor = False, normalize_embeddings = True)
            self.keys.append(query)
            self.index.add(embedding)
            self.cache[query] = {"embedding": embedding[0], "result": result}
            self.save_cache()
            logging.info("Query and Embedding added to Cache Successfully!")
            print(f"Cache After Append: {self.cache}")
        except Exception as e:
            logging.info("Cache Not Updated...")
            raise CustomException(e, sys)

    def retrieve_from_cache(self, query):
        try:
            embedding = self.model.encode([query], convert_to_tensor = False, normalize_embeddings = True)
            if len(self.keys) == 0:
                return None
            
            distances, indices = self.index.search(embedding, k = 1)
            if distances[0][0] <= (1 - self.similarity_threshold):
                closest_query = self.keys[indices[0][0]]
                return self.cache[closest_query]["result"]
            
            return None
        except Exception as e:
            raise CustomException(e, sys)
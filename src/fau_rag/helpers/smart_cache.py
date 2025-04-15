from rag.rag_retrieval import retriever
from helpers.exception import CustomException
from helpers.logger import logging
import faiss
from faiss import IndexFlatL2
import numpy as np
import sys
import json
import os

CACHE_FILE = "/src/fau_rag/config/smart_cache.json"

class SmartCache:
    def __init__(self, index: IndexFlatL2 = faiss.IndexFlatL2(128), model_name = retriever, similarity_threshold = 0.7, cache_file = CACHE_FILE):
        self.model = model_name
        self.similarity_threshold = similarity_threshold
        self.cache = {}
        self.index = index
        self.keys = []
        self.cache_file = cache_file
        self.load_cache()

    def save_cache(self):
        try:
            serializable_cache = { # Embeddings to lists before saving
                query: {
                    "embedding": data["embedding"].tolist() if isinstance(data["embedding"], np.ndarray) else data["embedding"],
                    "result": data["result"],
                }
                for query, data, in self.cache.items()
            }
            with open(self.cache_file, "w") as f:
                json.dump(serializable_cache, f, indent = 4)
            logging.info("Cache saved to file successfully.")
        except Exception as e:
            logging.error(f"Error saving cache to file: {e}")
            raise CustomException(e, sys) 
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    loaded_cache = json.load(f)
                
                self.cache = {
                    query: {
                        "embedding": np.array(data["embedding"]),
                        "result": data["result"],
                    }
                    for query, data in loaded_cache.items()
                }
                logging.info("Cache loaded from file successfully.")
            except Exception as e:
                logging.error(f"Error loading cache from file: {e}")
        else:
            logging.info(f"No cache file found at {self.cache_file}. Starting with an empty cache.")

    def get_cache_contents(self):
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
            # print(f"Cache After Append: {self.cache}")
        except Exception as e:
            logging.info("Cache Not Updated...")

    def retrieve_from_cache(self, query):
        try:
            self.load_cache()
            embedding = self.model.encode([query], convert_to_tensor = False, normalize_embeddings = True)
            if len(self.keys) == 0:
                print("No keys in cache.")
                return None
            
            distances, indices = self.index.search(embedding, k = 1)
            if len(indices[0]) == 0 or indices[0][0] >= len(self.keys):
                print("FAISS returned an invalid index. Cache and Index might be out of sync.")
                return None
            
            closest_distance = distances[0][0]
            closest_query = self.keys[indices[0][0]]

            # Debugging
            print(f"Query: {query}")
            print(f"Closest Query: {closest_query}")
            print(f"Closest Distance: {closest_distance}")

            if closest_distance <= (1 - self.similarity_threshold):
                closest_query = self.keys[indices[0][0]]
                print(f"Cache Hit!")
                return self.cache[closest_query]["result"]
            else:
                print("No similar query found in cache.")
                return None
        except Exception as e:
            raise CustomException(e, sys)

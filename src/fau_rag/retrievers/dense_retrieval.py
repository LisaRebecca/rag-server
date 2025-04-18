from sentence_transformers import SentenceTransformer
from helpers.exception import CustomException
from helpers.logger import logging
from helpers.utils import fast_rerank
from dataclasses import dataclass
from pathlib import Path

from constants import CONFIG_FILE_PATH
from helpers.utils import read_yaml

import sys
import time
import numpy as np

@dataclass(frozen = True)
class DenseRetrieverConfig:
    transformer: Path

class ConfigurationManager:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.dense_retriever_config = DenseRetrieverConfig(
            transformer = self.config["dense_retriever"]["transformer"]
        )

    def get_dense_retriever_config(self) -> DenseRetrieverConfig:
        return self.dense_retriever_config
    
config_manager = ConfigurationManager()
dense_retriever_config = config_manager.get_dense_retriever_config()

class DenseRetrieval:

    def __init__(self, config: DenseRetrieverConfig):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)

    async def dense_retrieval(query_embedding, index, metadata, top_k):
        try:
            start_time = time.time()  # Start timing

            _, indices = index.search(query_embedding, top_k)

            retrieved_docs = [metadata[i]["text"] for i in indices[0]]

            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time
            print(f"Dense retrieval took {elapsed_time:.4f} seconds")
            logging.info("Documents Retrieved Successfully! - Dense R")
            return retrieved_docs
        
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise CustomException(e, sys)
        
    def dense_retrieval_reranking(self, query_embedding, index, metadata, top_k):
        try:
            start_time = time.time()  # Start timing
            # Step 1: Encode the query and normalize it
            query_embed = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

            # Step 2: Retrieve the top_k nearest neighbors from the index
            _, indices = index.search(query_embed, top_k)
            
            # Step 3: Get candidate documents using the indices
            retrieved_docs = [{"text": metadata[i]["text"], "url": metadata[i]["url"]}
                            for i in indices[0]]
                        
            # Step 4: Extract candidate texts and compute their embeddings
            candidate_texts = [doc["text"] for doc in retrieved_docs]
            candidate_embeddings = self.retriever.encode(candidate_texts)
            candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            
            # Step 5: Re-rank the candidates using fast_rerank
            reranked_texts = fast_rerank(query_embed, candidate_embeddings, candidate_texts, final_top_k=top_k)
            
            # Step 6: Map the re-ranked texts back to full document metadata
            text_to_doc = {doc["text"]: doc for doc in retrieved_docs}
            reranked_docs = [text_to_doc[text] for text in reranked_texts]

            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time
            print(f"Dense retrieved and re-ranked successfully in {elapsed_time:.4f} seconds")
            logging.info("Documents retrieved and re-ranked successfully!")
            return reranked_docs

        except Exception as e:
            logging.error(f"Error during dense retrieval: {e}")
            return []
        
    def dense_embeddings(dense_model, metadata, batch_size = 32):
        try:
            if not metadata:
                raise ValueError("Metadata Empty or None!")
            
            for entry in metadata:
                if entry.get("text"):
                    texts = [entry.get("text")]

            if not texts:
                raise ValueError("No valid Text entries found in your Metadata!")
            
            print(f"Total valid texts to process: {len(texts)}")

            embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(total_batches):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start::batch_end]

                try:
                    print(f"Processing batch {i + 1}/{total_batches}...")
                    batch_embeddings = dense_model.encode(batch_texts, convert_to_numpy=True)
                    embeddings.extend(batch_embeddings)

                except Exception as e:
                    print(f"Error processing batch {i + 1}: {e}")
                    continue

            embeddings = np.array(embeddings)
            print(f"Computed embeddings for {embeddings.shape[0]} entries.")
            return embeddings
    
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise CustomException(e, sys)
        
dense_retrieval_instance = DenseRetrieval(dense_retriever_config)
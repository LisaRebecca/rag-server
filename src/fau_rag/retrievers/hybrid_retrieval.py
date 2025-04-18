from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from helpers.exception import CustomException
from sklearn.preprocessing import MinMaxScaler
from helpers.logger import logging
from helpers.utils import (preprocess_knowledgebase,
                           remove_duplicates)
from dataclasses import dataclass
from pathlib import Path

from constants import CONFIG_FILE_PATH
from helpers.utils import read_yaml
from retrievers.dense_retrieval import DenseRetrieval

import sys
import numpy as np

@dataclass(frozen = True)
class HybridRetrieverConfig:
    transformer: Path

class ConfigurationManager:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.hybrid_retriever_config = HybridRetrieverConfig(
            transformer = self.config["hybrid_retriever"]["transformer"]
        )

    def get_hybrid_retriever_config(self) -> HybridRetrieverConfig:
        return self.hybrid_retriever_config
    
config_manager = ConfigurationManager()
hybrid_retriever_config = config_manager.get_hybrid_retriever_config()

class HybridRetrieval:

    def __init__(self, config: HybridRetrieverConfig):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)
        self.bm25_model = None
        self.dense_embeddings = None

    def hybrid_preprocess(self, metadata):
        try:
            logging.info("Starting Hybrid Preprocessing...")
            # BM25 model
            tokenized_corpus = preprocess_knowledgebase(metadata)
            self.bm25_model = BM25Okapi(tokenized_corpus)
            logging.info("Sparse Model Completed Successfully!")

            # Compute dense embeddings
            logging.info("Starting Dense Embeddings..")
            self.dense_embeddings = DenseRetrieval.dense_embeddings(self.retriever, metadata)
            logging.info("Dense Embeddings Completed Successfully!")

        except Exception as e:
            raise CustomException(e, sys)
        
    def sparse_hybrid_retrieve(self, query, metadata, top_k = 20):
        tokenized_query = word_tokenize(query.lower())
        try:
            scores = self.bm25_model.get_scores(tokenized_query)
            logging.info("Sparse Scores Computed Successfully!")
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            top_k_results = [(idx, scores[idx]) for idx in top_k_indices]
            logging.info("Top K results returned successfully")
            return top_k_results
        except Exception as e:
            raise CustomException(e, sys)
        
    def dense_hybrid_retreive(self, query, embeddings, top_k = 20):
        if embeddings is None:
            raise ValueError("Embeddings not Found")
        try:
            query_embedding = self.dense_model.encode(query, convert_to_numpy=True)
            scores = np.dot(embeddings, query_embedding)
            logging.info("Dense Embeddings Computed Successfully!")

            top_indices = np.argsort(scores)[::-1][:top_k]
            dense_results = [(index, scores[index]) for index in top_indices if scores[index] > 0]  # Filter out zero scores
            return dense_results
        except Exception as e:
            raise CustomException(e, sys)
        
    def hybrid_retrieval(self, query, metadata, top_k, alpha):
        try:
            self.hybrid_preprocess(metadata)
            
            # Sparse retrieval
            sparse_results = self.sparse_hybrid_retrieve(query, metadata, top_k=top_k)
            # sparse_results = filter_results(sparse_results, query)

            # Dense retrieval
            dense_results = self.dense_hybrid_retreive(query, self.dense_embeddings, top_k=top_k)
            # dense_results = filter_results(dense_results, query)

            # Normalize scores
            scalar = MinMaxScaler()
            norm_dense_scores = scalar.fit_transform(np.array(dense_results).reshape(-1, 1)).flatten()
            norm_sparse_scores = scalar.fit_transform(np.array(sparse_results).reshape(-1, 1)).flatten()
            logging.info("Sparse and Dense Scores Normalized Successfully!")

            # Combine Scores
            scores = {}
            for i, (index, _) in enumerate(sparse_results):
                scores[index] = scores.get(index, 0) + alpha * norm_sparse_scores[i]

            for i, (index, _) in enumerate(dense_results):
                scores[index] = scores.get(index, 0) + (1 - alpha) * norm_dense_scores[i]
            
            logging.info("Scores Combined Successfully!")
            
            # Retrieve Top Results
            top_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                               
            results = [metadata[i] for i, _ in top_indices]
            results = remove_duplicates(results)
            for idx, result in enumerate(results, 1):
                print(f"Result {idx}:")
                print(f"- **Text:** {result['text']}")
                print(f"- **URL:** {result['url']}\n")

            return results
            
        except Exception as e:
            raise CustomException(e, sys)
        
hybrid_retrieval_instance = HybridRetrieval(hybrid_retriever_config)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.utils import preprocess_knowledgebase
from dataclasses import dataclass
from pathlib import Path

from constants import CONFIG_FILE_PATH
from helpers.utils import read_yaml

import sys
import numpy as np

@dataclass(frozen = True)
class SparseRetrieverConfig:
    transformer: Path

class ConfigurationManager:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.sparse_retriever_config = SparseRetrieverConfig(
            transformer = self.config["sparse_retriever"]["transformer"]
        )

    def get_sparse_retriever_config(self) -> SparseRetrieverConfig:
        return self.sparse_retriever_config
    
config_manager = ConfigurationManager()
sparse_retriever_config = config_manager.get_sparse_retriever_config()

class SparseRetrieval:

    def __init__(self, config: SparseRetrieverConfig):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)

    def tokenized_corpus(metadata):
        return preprocess_knowledgebase(metadata)
        
    def sparse_retrieval(query, metadata, top_k):
        try:
            # Prep corpus
            tokenized_corpus = preprocess_knowledgebase(metadata)

            # Initialize BM25
            bm25 = BM25Okapi(tokenized_corpus)

            # Query processing
            tokenized_query = word_tokenize(query.lower())
            scores = bm25.get_scores(tokenized_query)

            # Get top-k docs
            # top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            top_indices = np.argsort(scores)[::-1][:top_k]
            retrieved_docs = [metadata[i] for i in top_indices]

            logging.info("Documents Retrieved Successfully! - Sparese R")
            print("Retrieved Documents Sparse: ", retrieved_docs)
            for idx, result in enumerate(retrieved_docs, 1):
                return print(f"Result {idx}:\nText: {result['text']}\nURL: {result['url']}\n")
            
        except Exception as e:
            raise CustomException(e, sys)
        
sparse_retrieval_instance = SparseRetrieval(sparse_retriever_config)
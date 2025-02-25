from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.utils import preprocess_knowledgebase, dense_embeddings, remove_duplicates, filter_results
from sklearn.preprocessing import MinMaxScaler

import sys
import time
import numpy as np

import httpx

retriever = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # Hugging Face model

class RAG_Retrieval:

    def __init__(self):
        self.bm25_model = None
        self.dense_model = retriever
        self.dense_embeddings = None

    def cosine_similarity(a, b):
        dot_product = sum([x * y for x, y in zip(a,b)])
        norm_a = sum([x ** 2 for x in a]) ** 0.5
        norm_b = sum([x ** 2 for x in b]) ** 0.5
        return dot_product / (norm_a * norm_b)
    
    def tokenized_corpus(metadata):
        return preprocess_knowledgebase(metadata)

    async def dense_retrieval(query_embedding, index, metadata, top_k):
        try:
            start_time = time.time()  # Start timing
            # query_embedding = retriever.encode([query])

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
        
    def sparse_retrieval(query, metadata, top_k):
        try:
            # import nltk
            # nltk.download('punkt_tab')

            # Prep corpus
            # corpus = [entry["text"] for entry in metadata]
            # tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
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
        
    def hybrid_preprocess(self, metadata):
        try:
            logging.info("Starting Hybrid Preprocessing...")
            # BM25 model
            tokenized_corpus = preprocess_knowledgebase(metadata)
            self.bm25_model = BM25Okapi(tokenized_corpus)
            logging.info("Sparse Model Completed Successfully!")

            # Compute dense embeddings
            logging.info("Starting Dense Embeddings..")
            self.dense_embeddings = dense_embeddings(self.dense_model, metadata)
            logging.info("Dense Embeddings Completed Successfully!")

        except Exception as e:
            raise CustomException(e, sys)
        
    def sparse_retrieve(self, query, metadata, top_k = 20):
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
        
    def dense_retreive(self, query, embeddings, top_k = 20):
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
            sparse_results = self.sparse_retrieve(query, metadata, top_k=top_k)
            # sparse_results = filter_results(sparse_results, query)

            # Dense retrieval
            dense_results = self.dense_retreive(query, self.dense_embeddings, top_k=top_k)
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
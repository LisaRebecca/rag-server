from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.utils import preprocess_knowledgebase, dense_embeddings
from sklearn.preprocessing import MinMaxScaler

import sys
import numpy as np

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

    def dense_retrieval(query, index, metadata, top_k):
        try:
            query_embedding = retriever.encode([query])

            _, indices = index.search(np.array(query_embedding, dtype = "float32"), top_k) # Nearest Neighbor

            retrieved_docs = [metadata[i]["text"] for i in indices[0]]

            # print("Retrieved Documents:", retrieved_docs)

            logging.info("Documents Retrieved Successfully! - Dense R")
            print("Retrieved Documents Dense: ",retrieved_docs)
            return retrieved_docs

        except Exception as e:
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

            print("bm25 model:", self.bm25_model)
            logging.info("Sparse Model Completed Successfully!")

            # Compute dense embeddings
            logging.info("Starting Dense Embeddings..")
            dense_embeddings(self.dense_model, metadata)
            print("dense embeddings:", self.dense_embeddings)
            logging.info("Dense Embeddings Completed Successfully!")

        except Exception as e:
            raise CustomException(e, sys)
        
    def sparse_retrieve(self, query, top_k = 20):
        tokenized_query = word_tokenize(query.lower())    
        scores = self.bm25_model.get_scores(tokenized_query.split())
        logging.info("Sparse Scores Computed Successfully!")

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(index, scores[index]) for index in top_indices]

    def dense_retreive(self, query, embeddings, top_k = 20):
        query_embedding = self.dense_model.encode(query, convert_to_numpy=True)
        scores = np.dot(embeddings, query_embedding)
        logging.info("Dense Embeddings Computed Successfully!")

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(index, scores[index]) for index in top_indices]
    
    def hybrid_retrieval(self, query, metadata, top_k, alpha):
        try:
            preprocessed_metadata = self.hybrid_preprocess(metadata)
            logging.info("Metadata Preprocessed Successfully!")
            
            # Sparse retrieval
            sparse_results = self.sparse_retrieve(query, top_k=top_k)
            logging.info("Sparse Scores: ", sparse_results)

            # Dense retrieval
            dense_results = self.dense_retreive(query, self.dense_embeddings, top_k=top_k)

            # Get sparse and dense scores
            # dense_scores = RAG_Retrieval.dense_retreive(query, dense_embeddings)
            # sparse_scores = RAG_Retrieval.sparse_retrieve(query, sparse_model, metadata)

            logging.info("Dense Scores: ", dense_results)
            # Normalize scores
            scalar = MinMaxScaler()
            norm_dense_scores = scalar.fit_transform(dense_results.reshape(-1,-1)).flatten()
            norm_sparse_scores = scalar.fit_transform(sparse_results.reshape(-1,-1)).flatten()

            # Combine Scores
            # combined_scores = alpha * norm_dense_scores + (1-alpha) * norm_sparse_scores
            # top_indices = np.argsort(combined_scores)[::-1][:top_k]
            scores = {}
            for index, score in norm_sparse_scores:
                scores[index] = scores.get(index, 0) + alpha * score
            for index, score in norm_dense_scores:
                scores[index] = scores.get(index, 0) + (1 - alpha) * score

            top_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # Retrieve Top Results
            results = [preprocessed_metadata[i] for i in top_indices]
            for idx, result in enumerate(results, 1):
                return print(f"Result {idx}:\nText: {result['text']}\nURL: {result['url']}\n")
            
        except Exception as e:
            raise CustomException(e, sys)
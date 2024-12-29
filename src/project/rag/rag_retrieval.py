from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from helpers.exception import CustomException
from helpers.logger import logging
from helpers.utils import preprocess_knowledgebase

import sys
import json
import numpy as np

retriever = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # Hugging Face model

class RAG_Retrieval:

    def cosine_similarity(a, b):
        dot_product = sum([x * y for x, y in zip(a,b)])
        norm_a = sum([x ** 2 for x in a]) ** 0.5
        norm_b = sum([x ** 2 for x in b]) ** 0.5
        return dot_product / (norm_a * norm_b)

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
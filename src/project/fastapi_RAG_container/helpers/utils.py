import os
import sys
import json
import numpy as np
import faiss
from helpers.exception import CustomException
from helpers.logger import logging
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from fastapi import APIRouter, Depends, HTTPException, status, Header
from typing import Optional, List



def preprocess_knowledgebase(knowledgebase):
    corpus = []
    for entry in knowledgebase:
        tokens = word_tokenize(entry["text"].lower())
        corpus.append(tokens)
    return corpus

def remove_duplicates(results):
    unique_texts = set()
    unique_results = []
    for result in results:
        if result["text"] not in unique_texts:
            unique_results.append(result)
            unique_texts.add(result["text"])
    return unique_results

def filter_results(results, query):
    filtered_results = []
    keywords = query.lower().split()
    for result in results:
        if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict) and 'text' in result[0]:
            metadata = result[0]
            for keyword in keywords:
                if any(keyword in metadata['text'].lower()):
                    filtered_results.append(result)
    return filtered_results

def dense_embeddings(dense_model, metadata, batch_size = 32):
    if not metadata:
        raise ValueError("Metadata Empty or None!")
    
    #texts = [entry.get("text") for entry in metadata if entry.get("texts")]
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
    
def save_vector_db(embeddings, chunks, index_file, metadata_file):
    try:
        # Save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        faiss.write_index(index, index_file)

        # Save metadata as JSONL
        with open(metadata_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        
        logging.info("Data Saved To Vector DB (FAISS and JSONL)!")
    except Exception as e:
        raise CustomException(e, sys)

def load_vector_db(index_file, metadata_file):
    try:
        # Check if the FAISS index file exists
        if not os.path.exists(index_file):
            print(f"FAISS index not found at {index_file}. Creating a new one...")
            create_vector_db_from_jsonl(metadata_file, index_file)

        # Load FAISS index
        print("Loading FAISS index...")
        index = faiss.read_index(index_file)

        # Load metadata from JSONL
        metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                metadata.append(json.loads(line.strip()))
        
        logging.info("Data Loaded From Vector DB Successfully (FAISS and JSONL)!")
        return index, metadata
    except Exception as e:
        raise CustomException(e, sys)

def create_vector_db_from_jsonl(metadata_file, index_file):
    try:
        # Load metadata from JSONL
        metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())  # Parse each line as JSON
                    if isinstance(entry, list):  # Handle unexpected nested lists
                        metadata.extend(entry)
                    elif isinstance(entry, dict):  # Normal case
                        metadata.append(entry)
                    else:
                        print(f"Skipping unexpected entry type: {type(entry)}")
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {line.strip()} (Error: {e})")

        # Validate metadata structure
        if not all(isinstance(entry, dict) and "text" in entry for entry in metadata):
            raise ValueError("Metadata entries must be dictionaries with a 'text' key.")

        # Initialize embedding model
        print("Initializing embedding model...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Generate embeddings for each chunk's text
        print("Generating embeddings for metadata...")
        embeddings = []
        for entry in metadata:
            try:
                embeddings.append(model.encode(entry["text"]))
            except KeyError:
                print(f"Skipping entry without 'text': {entry}")

        embeddings = np.array(embeddings, dtype="float32")  # Convert to NumPy array

        # Create FAISS index
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        # Save FAISS index
        print(f"Saving FAISS index to {index_file}...")
        faiss.write_index(index, index_file)

        print("FAISS index created and saved successfully.")

    except Exception as e:
        raise RuntimeError(f"Error creating FAISS index: {e}")

def verify_api_key(
    authorization: Optional[str] = Header(None), 
    API_KEYS: list[str] = []
):
    if not isinstance(authorization, str):
        raise ValueError("Authorization must be provided as a string.")

    if authorization is None or not authorization.startswith("Bearer "):
        logging.warning("No authorization header provided or malformed header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ")[1]
    if token not in API_KEYS:
        logging.warning(f"Invalid API key attempted: {token}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

def load_all_model_configs(config_path: str):
   
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at path: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    return configs

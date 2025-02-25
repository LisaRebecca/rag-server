import sys
from sentence_transformers import SentenceTransformer
from helpers.exception import CustomException
from pydantic import BaseModel
from dataclasses import dataclass
from pathlib import Path
from fastapi import HTTPException
import uvicorn
import requests
import numpy as np

from fastapi import FastAPI

from constants import CONFIG_FILE_PATH
from helpers.utils import read_yaml

from openai import OpenAI
from typing import List

app = FastAPI()

@dataclass(frozen = True)
class EmbeddingModelConfig:
    transformer: Path
    openai_api_key: str
    openai_api_base: str

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str

class EmbeddingData(BaseModel):
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]

class ConfigurationManager:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.embedding_model_config = EmbeddingModelConfig(
            transformer = self.config["embedding_model"]["transformer"],
            openai_api_key = self.config["embedding_model"]["openai_api_key"],
            openai_api_base = self.config["embedding_model"]["openai_api_base"]
        )

    def get_embedding_model_config(self) -> EmbeddingModelConfig:
        return self.embedding_model_config
    
config_manager = ConfigurationManager()
embedding_model_config = config_manager.get_embedding_model_config()
    
class EmbeddingModel:
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)
        self.client = OpenAI(
            api_key = self.config.openai_api_key,
            base_url = self.config.openai_api_base
        )
        
    def get_embedding(self, query: str):
        try:
            url = self.config.openai_api_base
            model = self.retriever
            payload = {"input": [query], "model": str(model)}
            
            response = requests.post(url, json = payload)
            response.raise_for_status()

            embedding = response.json()["data"][0]["embedding"]
            print(f"Embedding generated successfully. For query: {query}, Length: {len(embedding)}")

            return embedding
        
        except Exception as e:
            raise CustomException(e, sys)
        
embedding_model_instance = EmbeddingModel(embedding_model_config)

@app.post("/v1/embeddings", response_model = EmbeddingResponse)
async def openai_compliant_embedding(request: EmbeddingRequest):
    """
    OpenAI API compliant endpoint to generate embeddings for a given text input.
    """
    try:
        embeddings = embedding_model_instance.retriever.encode(request.input, convert_to_numpy = True)

        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        response = EmbeddingResponse(
            data = [EmbeddingData(embedding = embed) for embed in embeddings_list]
        )

        print(f"response: {response}")
        return response
    
    except Exception as e:
            print(f"Error generating embedding: {e}")
            raise HTTPException(status_code = 500, detail = str(e))
    
if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="127.0.0.1",
        port=8070,
        log_level="info",
    )
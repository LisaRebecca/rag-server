import requests
# import aioredis
import sys
from helpers.exception import CustomException

class RAG:
    def __init__(self):
        self.retrieval_url = (
            "https://example.com/retrieve"  # Replace with actual retrieval system URL
        )
        self.generation_url = (
            "https://example.com/generate"  # Replace with actual generation system URL
        )

    def retrieve(self, query):
        """
        Retrieve relevant documents from the retrieval system based on the query.
        """
        try:
            response = requests.post(self.retrieval_url, json={"query": query})
            response.raise_for_status()
            return response.json()["documents"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Retrieval failed: {str(e)}")
        
    # async def inspect_cache(self):
    #     """
    #     Retrieve and print all keys and their values currently in the Redis cache.
    #     """
    #     Cache_Contents = {}
    #     try:
    #         Redis = await aioredis.from_url("redis://redis")
    #         Keys = await Redis.keys("*")
    #         if not Keys:
    #             print("Cache is currently Empty")
                
    #         for Key in Keys:
    #             Value = await Redis.get(Key)
    #             if isinstance(Value, bytes):
    #                 Cache_Contents[Key] = Value.decode("utf-8")
    #             else:
    #                 Value
    #         await Redis.close()

    #         return Cache_Contents
        
    #     except Exception as e:
    #         raise CustomException(e, sys)
        
    def generate(self, query):
        """
        Generate an augmented response using the retrieved documents and the query.
        """
        return "Dummy Result, pls implement RAG :)"
        # Retrieve relevant documents
        documents = self.retrieve(query)

        # Combine query and documents for the generation task
        input_data = {"query": query, "documents": documents}

        try:
            response = requests.post(self.generation_url, json=input_data)
            response.raise_for_status()
            return response.json()["generated_text"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Generation failed: {str(e)}")
        

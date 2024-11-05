import requests
import httpx
import redis
import aioredis
import json
import hashlib
import os

class RAG:
    def __init__(self): # Defining Endpoints - Need to replace with actual endpoints
        self.redis_client = aioredis.from_url("redis://redis", encoding="utf-8", decode_responses=True)
        self.retrieval_url = "https://jsonplaceholder.typicode.com/posts"
        self.generation_url = "https://jsonplaceholder.typicode.com/comments"

    def generate_cache_key(self, query):
        """
        Generate unique hash key based on the query.
        """
        return hashlib.md5(query.encode()).hexdigest()

    async def retrieve(self, query):
        """
        Asynchronously retrieve relevent documents from the retrieval system using `async` and `await`.
        """
        try:
            # Response = requests.post(self.retrieval_url, json = {"query": query})
            Cache_Key = self.generate_cache_key(f"retrieve:{query}")
            Cached_Data = await self.redis_client.get(Cache_Key)
            if Cached_Data:
                return json.loads(Cached_Data)
            async with httpx.AsyncClient() as Client:
                # Response = requests.post(self.retrieval_url)
                Response = await Client.get(self.retrieval_url)
                Response.raise_for_status()
                Posts = Response.json()
                if isinstance (Posts, list):
                    Documents = [Post['title'] for Post in Posts[:2]]
                else:
                    raise Exception("Expected a list of Posts, but got something else.")
                await self.redis_client.setex(Cache_Key, 300, json.dumps(Documents))
                return Documents
        # except requests.exceptions.RequestException as e:
        except httpx.RequestError as e:
            raise Exception(f"Retrieval Failed: {str(e)}")
        
    async def generate(self, query):
        """
        Asynchronously generate a response using the retrieved documents and query.
        """
        # Documents = self.retrieve(query)
        Cache_Key = self.generate_cache_key(f"generate:{query}")
        Cached_Data = await self.redis_client.get(Cache_Key)
        if Cached_Data:
            return json.loads(Cached_Data)
        # Documents = await self.retrieve(query)
        # Input_Data = {"query": query, "documents": Documents}
        try:
            # Response = requests.post(self.generation_url, json = Input_Data)
            async with httpx.AsyncClient() as Client:
                Response = await Client.get(self.generation_url)
                Response.raise_for_status()
                Generated_Text = Response.json()[0]['body']
                await self.redis_client.setex(Cache_Key, 300, json.dumps(Generated_Text))
                return Generated_Text
        # except requests.exceptions.RequestException as e:
        except httpx.RequestError as e:
            raise Exception(f"Generation Failed: {str(e)}")
        
    async def inspect_cache(self):
        """
        Retrieve and print all keys and their values currently in the Redis cache.
        """
        Cache_Contents = {}
        try:
            Redis = await aioredis.from_url("redis://redis")
            Keys = await Redis.keys("*")
            if not Keys:
                print("Cache is currently Empty")
                
            for Key in Keys:
                Value = await Redis.get(Key)
                if isinstance(Value, bytes):
                    Cache_Contents[Key] = Value.decode("utf-8")
                else:
                    Value
            await Redis.close()
        except Exception as e:
            return {"Error": str(e)}
        return Cache_Contents
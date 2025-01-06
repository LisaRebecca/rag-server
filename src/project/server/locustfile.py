from locust import HttpUser, TaskSet, task, between
import random
from pydantic import BaseModel
from typing import Dict, List

QUERIES = [
    "Tell me about optimization courses at the FAU",
    "What are the prerequisites for the data science module?",
    "How do I get into the FAU?",
    "What is the RRZE?",
    "What are the benefits of studying computer science at FAU?",
    "How many ECTS do I need to graduate from a data science masters program?"
]

class RAGUserBehavior(TaskSet):
    # @task(1)
    # def query_rag(self):
    #     self.client.post(
    #         "/api/query",
    #         json = {"query": "Tell me about optimization courses at the FAU"}
    #     )

    # class QueryPayload(BaseModel):
    #     queries: List[str]

    @task
    def concurrent_query(self):
        query = random.choice(QUERIES)
        payload = {"query": query}

        response = self.client.post("/api/query", json = payload)

        print(f"Sent Query: {query}")
        print(f"Response: {response.status_code}, {response.text}")

    # @task(2)
    # def health_check(self):
    #     self.client.get("/health")

    # @task(3)
    # def metrics_check(self):
    #     self.client.get("/metrics")

class RAGUser(HttpUser):
    tasks = [RAGUserBehavior]
    wait_time = between(0.5,1.5) # Random delays between requests
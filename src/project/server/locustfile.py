from locust import HttpUser, TaskSet, task, between

class RAGUserBehavior(TaskSet):
    @task(1)
    def query_rag(self):
        self.client.post(
            "/api/query",
            # json = {"query": "What are the prerequisites to the Data Science Module?"}
            # json = {"query": "How do I enroll into FAU?"}
            json = {"query": "Tell me about optimization courses at the FAU"}
        )

    @task(2)
    def health_check(self):
        self.client.get("/health")

    @task(3)
    def metrics_check(self):
        self.client.get("/metrics")

class RAGUser(HttpUser):
    tasks = [RAGUserBehavior]
    wait_time = between(1,5) # Random delays between requests
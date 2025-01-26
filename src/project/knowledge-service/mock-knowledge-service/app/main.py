from fastapi import FastAPI, Query
from typing import List

from common.api.base_service import BaseKnowledgeService, KnowledgeItem, SearchResponse

app = FastAPI(title="Memory Knowledge Service", version="1.0")

# Example data (in-memory store)
MEMORY_KNOWLEDGE_ITEMS = [
    KnowledgeItem(
        id=1,
        title="Introduction to FastAPI",
        content="FastAPI is a modern web framework for Python.",
        source="https://fastapi.tiangolo.com/",
        score=0.0  # Placeholder
    ),
    KnowledgeItem(
        id=2,
        title="Understanding REST APIs",
        content="REST APIs are a popular way to build web services.",
        source="https://example.com/rest-apis",
        score=0.0
    ),
]

class MemoryKnowledgeService(BaseKnowledgeService):
    def fetch_knowledge_items(self) -> List[KnowledgeItem]:
        return MEMORY_KNOWLEDGE_ITEMS

# Instantiate the service
memory_service = MemoryKnowledgeService()

@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_knowledge(
    query: str = Query(..., description="Search term to filter knowledge items"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of results to return (1-100)"),
):
    return memory_service.search_knowledge(query, max_results)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint to ensure the service is running.
    """
    return {"status": "ok"}
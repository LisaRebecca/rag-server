from pydantic import BaseModel

class KnowledgeRequest(BaseModel):
    query: str
    limit: int = 10

class KnowledgeItem(BaseModel):
    id: int
    title: str
    content: str
    source: str
    score: float

class SearchResponse(BaseModel):
    items: list[KnowledgeItem]
    total: int

from pydantic import BaseModel
from typing import Optional, Literal

search_modes = Literal["keyword", "similarity", "hybrid"]

class KnowledgeRequest(BaseModel):
    query: str
    limit: int = 10
    search_mode: Optional[search_modes] = None

class KnowledgeItem(BaseModel):
    id: int
    title: str
    content: str
    source: str
    score: float
    search_mode: Optional[search_modes] = None

class SearchResponse(BaseModel):
    items: list[KnowledgeItem]
    total: int

from abc import ABC, abstractmethod
from pydantic import BaseModel, HttpUrl
from typing import List

from common.api.models import KnowledgeItem, KnowledgeRequest, SearchResponse

# Abstract Base Class
class BaseKnowledgeService(ABC):
    @abstractmethod
    def fetch_knowledge_items(self) -> List[KnowledgeItem]:
        """
        Abstract method to fetch knowledge items.
        Concrete implementations must override this method.
        """
        pass

    def calculate_similarity(self, query: str, text: str) -> float:
        """
        Calculate a similarity score between the query and the text.

        Args:
            query (str): The search query.
            text (str): The text to compare against.

        Returns:
            float: A similarity score between 0 and 1.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, query.lower(), text.lower()).ratio()

    def search_knowledge(self, query: str, max_results: int) -> SearchResponse:
        """
        Search for knowledge items by a query term, limited to a maximum number of results.

        Args:
            query (str): The search term to filter knowledge items.
            max_results (int): The maximum number of results to return.

        Returns:
            SearchResponse: A list of matching knowledge items with similarity scores and the total count.
        """
        knowledge_items = self.fetch_knowledge_items()

        # Find matches and calculate similarity scores
        matching_items = [
            KnowledgeItem(
                id=item.id,
                title=item.title,
                content=item.content,
                source=item.source,
                score=max(
                    self.calculate_similarity(query, item.title),
                    self.calculate_similarity(query, item.content)
                )
            )
            for item in knowledge_items
            if query.lower() in item.title.lower() or query.lower() in item.content.lower()
        ]

        # Sort by score in descending order
        matching_items.sort(key=lambda x: x.score, reverse=True)

        # Limit results
        limited_items = matching_items[:max_results]

        return SearchResponse(items=limited_items, total=len(matching_items))

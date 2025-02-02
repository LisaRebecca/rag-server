from fastapi import FastAPI, Query

from common.api.base_service import BaseKnowledgeService
from document_knowledge_service.app.document_service import DocumentKnowledgeService

app = FastAPI()

# Initialize the document knowledge service
pdf_folder = "./example-data/ai-examination-regulations"
doc_service = DocumentKnowledgeService(pdf_folder)

class DocumentKnowledgeAPI(BaseKnowledgeService):
    def __init__(self, service: DocumentKnowledgeService):
        self.service = service

    def search(self, query: str, max_results: int = 5):
        return self.service.search(query, max_results)
    
    def fetch_knowledge_items(self):
        pass

# Create an instance of the API
document_api = DocumentKnowledgeAPI(doc_service)

@app.get("/search")
async def search(query: str = Query(...), max_results: int = Query(5, ge=1, le=50)):
    """Search for passages matching the query."""
    results = document_api.search(query, max_results)
    return [{"passage": result[0], "score": result[1]} for result in results]

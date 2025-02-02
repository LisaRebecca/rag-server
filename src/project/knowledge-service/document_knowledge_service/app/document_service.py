import os
import hashlib
import time
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from itertools import islice

from common.api.models import KnowledgeRequest, KnowledgeItem, SearchResponse

class DocumentKnowledgeService:
    def __init__(self, pdf_folder: str, collection_name: str = "documents"):
        self.pdf_folder = pdf_folder
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )
        self._load_pdfs()

    def _load_pdfs(self):
        """Load and process PDFs from the folder."""
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, filename)
                self._process_pdf(pdf_path)

    def _process_pdf(self, pdf_path: str):
        """Extract text from a PDF and create embeddings with metadata."""
        reader = PdfReader(pdf_path)
        doc_id = os.path.basename(pdf_path)
        timestamp = time.time()
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                chunks = self._split_text(text, chunk_size=200)
                for j, chunk in enumerate(chunks):
                    passage_id = f"{doc_id}_page{i+1}_chunk{j+1}"
                    content_hash = hashlib.sha256(chunk.encode()).hexdigest()
                    metadata = {
                        "file_location": pdf_path,
                        "page_number": i + 1,
                        "timestamp": timestamp,
                        "content_hash": content_hash,
                    }
                    self.collection.add(
                        ids=[passage_id],
                        documents=[chunk],
                        metadatas=[metadata]
                    )

    def _split_text(self, text: str, chunk_size: int = 200) -> List[str]:
        """Split text into chunks of approximately `chunk_size` words."""
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def search(self, requests: List[KnowledgeRequest]) -> List[SearchResponse]:
        """Perform batch search and return multiple search responses."""
        query_texts = [req.query for req in requests]
        n_results = [req.limit for req in requests]
        batch_results = self.collection.query(query_texts=query_texts, n_results=max(n_results))

        responses = []
        for req, docs, scores, metas in zip(requests, batch_results["documents"], batch_results["distances"], batch_results["metadatas"]):
            items = [
                KnowledgeItem(
                    id=i,
                    title=metadata.get("file_location", "Unknown"),
                    content=doc,
                    source=metadata.get("file_location", "Unknown"),
                    score=score,
                    search_mode=req.search_mode
                )
                for i, (doc, score, metadata) in enumerate(zip(docs, scores, metas))
            ]
            responses.append(SearchResponse(items=items[:req.limit], total=len(items[:req.limit])))
        return responses

    def show_all_contents(self):
        """Retrieve and display all contents in ChromaDB."""
        return self.collection.get()
        

# Example usage
if __name__ == "__main__":
    pdf_folder = "./pdfs"
    service = DocumentKnowledgeService(pdf_folder)
    request = KnowledgeRequest(query="What is the impact of AI on finance?", limit=5)
    response = service.search(request)
    for item in response.items:
        print(f"ID: {item.id}\nTitle: {item.title}\nScore: {item.score:.4f}\nContent: {item.content}\nSource: {item.source}\nSearch Mode: {item.search_mode}\n")
    
    print("\nAll contents in ChromaDB:")
    service.show_all_contents()

import os
import pytest
from fastapi.testclient import TestClient
from document_knowledge_service.app.document_service import DocumentKnowledgeService
from document_knowledge_service.app.main import app

# Create a TestClient for FastAPI
service = DocumentKnowledgeService(pdf_folder="document_knowledge_service/tests/test_pdfs")
client = TestClient(app)

# Test PDFs directory
TEST_PDFS_DIR = "document_knowledge_service/tests/test_pdfs"

# test if there can be duplicate pdf chunks (post should be idempotent)

# ensure that the configuration (embedding model, chunk size etc can not be changed without creating a new service)
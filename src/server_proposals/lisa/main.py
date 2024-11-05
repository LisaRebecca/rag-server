from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import RAG

# FastAPI app
app = FastAPI(
    title="RAG API",
    description="A FastAPI server for a RAG application",
    version="1.0.0",
)

# Initialize RAG application
rag_app = RAG()

# Request model
class QueryRequest(BaseModel):
    query: str

# Response model
class QueryResponse(BaseModel):
    result: str

@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Endpoint for querying the RAG application.
    """
    try:
        result = rag_app.generate(request.query)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK"}

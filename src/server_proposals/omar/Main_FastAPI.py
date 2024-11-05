from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG import RAG
from Auth import Authenticate_User, status, timedelta, ACCESS_TOKEN_EXPIRE_MINUTES, Create_Access_Token, User, Depends, Get_Current_User

app = FastAPI(
    title = "RAG API",
    description = "A FastAPI server for a RAG application",
    version = "1.0.0"
)

rag_app = RAG()

# Pydantic Models for request and response validation:
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class TokenRequest(BaseModel):
    username: str
    password: str

# Endpoints
@app.post("/rag/query", response_model = QueryResponse)
async def Query_RAG(request: QueryRequest):
    try:
        Result = await rag_app.generate(request.query)
        return QueryResponse(response = Result)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Error Processing The Request: {str(e)}")
    
@app.get("/health")
async def Health_Status():
    return {"status": "OK"}

@app.get("/cache")
async def Cache_Contents():
    try:
        Cache_Contents = await rag_app.inspect_cache()
        return {"Cache Contents: ": Cache_Contents}
    except Exception as e:
        return {"Error": str(e)}
   
@app.post("/token")
async def Login_For_Access_Token(request: TokenRequest):
    User = await Authenticate_User(request.username, request.password)
    if not User:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect username or password",
        )
    Access_Token_Expires = timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    Access_Token = await Create_Access_Token(
        data = {"sub": User.username}, expires_delta = Access_Token_Expires
    )
    return {"Access_Token": Access_Token, "token_type": "bearer"}
    
@app.get("/secure-data")
async def Secure_Data(current_user: User = Depends(Get_Current_User)):
    return {"message": f"Hello, {current_user.username}! This is Protected Data..."}
from fastapi import FastAPI
from bs4 import BeautifulSoup
import requests
import openai

openai.api_key = "???"

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG app FastAPI Server!"}

@app.get("/retrieve/") #Endpoint
def retrieve_docs(query: str):
    URL = f"https://www.fau.eu/search?q={query}"

    Response = requests.get(URL)
    if Response.status_code == 200:
        Soup = BeautifulSoup(Response.content, "html.parser")

        Docs = [Tag.text for Tag in Soup.find_all("p")]
    return {"query": query, "documents": Docs}

@app.post("/generate/") #Endpoint
def generate_response(query: str):
    Docs = ["Doc1", "Doc2"]
    Prompt = f"Based on the following documents: {Docs}. Answer the quesr {query}"
    Response = openai.completions.create(
        engine = "text-davinci-003",
        prompt = Prompt,
        max_tokens = 100
    )

    Generated_Text = Response.choices[0].text.strip()
    return {"query": query, "response": Generated_Text}
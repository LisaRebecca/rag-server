## Welcome to **Tier0** our FAU RAG LLM

This project presents a **modular Retrieval-Augmented Generation (RAG) system** designed to optimize the balance between retrieval speed and response accuracy. By integrating flexible retrieval pipelines, this system enables efficient, scalable, and accurate AI-powered information querying — ideal for academic and enterprise environments.

## 🧰 Tech Stack

- 🐍 **Python 3.x**
- ⚡ **FastAPI**
- 🔗 **OpenAPI Specification**
- 🐳 **Docker**
- 🌐 **Redis** *(optional for caching)* 
- 🧪 **Locust** *(load testing)*
- 💻 **Open WebUI** *(frontend integration)*

## 1- Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/LisaRebecca/rag-server.git
   cd rag-server

## 2 - Knowledge Base
To download our chunked jsonl knowledge base goto [Knowledge Base](https://drive.google.com/file/d/1_4BNVhkEaAOngTAsgLgh38kWe0aQdrqW/view?usp=drive_link) 

## 3- FAISS Indices - Models
### [Light Index](https://drive.google.com/file/d/1qOECFQ_Df_sBCextiqRbPjeKHTFXpbdW/view?usp=sharing) (384 dims, all-MiniLM-L6-v2)
### [Normal Index](https://drive.google.com/file/d/1-0ncb5rZ-9SSosAocHnuR6iYIfLLdtNE/view?usp=sharing) (768 dims, all-mpnet-base-v2)

Update `VECTORSTORE_PATH` in `server/fastapi_router.py` and run your model.

## 4- Connecting to FAU endpoints - CISCO VPN
### Step 1: Install CISCO AnyConnect Client
Connect to ```vpn.fau.de```

### Step 2: Login using ```Idm``` username and password


## 5- Running The Service Using Docker
### Step 1: Open Docker app on your local machine

### Step 2: From your CMD in the project directory

For **first time** build run:
`docker build -f embedding_service/Dockerfile -t embedding-service .` for the **Embedding Service**

and

`docker build -f fastapi_RAG_service/Dockerfile -t fastapi-rag-app .` **FastAPI RAG App**

For **existing build** run `docker-compose up -d`

### Step 3: For OpenWebUI installation for OpenAI API Usage Only
`docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main`

## 💡 Usage

Once the FAU RAG service is running, goto Open WebUI Frontend at ```localhost:3000```
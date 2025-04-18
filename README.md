# 🚀 Welcome to **Tier0** — FAU's RAG LLM System

Tier0 is a **modular Retrieval-Augmented Generation (RAG) system** crafted to optimize the trade-off between **retrieval speed** and **response accuracy**. Designed with flexibility and scalability in mind, this system empowers AI-driven information querying — perfect for both academic and enterprise-grade environments.

---

## 🧠 Key Features

- **Fast & Accurate** — balances retrieval speed and relevance for optimal AI responses.
- **Modular Architecture** — easy to extend, swap, and maintain components.
- **Enterprise-Ready** — integrates smoothly into secure networks like FAU's infrastructure.
- **Flexible Frontend** — Open WebUI compatibility for user-friendly interaction.

---

## 🧰 Tech Stack Overview

| Tool/Service        | Purpose                                   |
|----------------------|-------------------------------------------|
| 🐍 **Python 3.x**        | Core backend logic                      |
| ⚡ **FastAPI**           | High-performance web server             |
| 🔗 **OpenAPI Spec**      | API standardization                     |
| 🐳 **Docker**            | Containerization for deployment         |
| 🌐 **Redis** *(optional)*| Smart caching for faster responses      |
| 🧪 **Locust**            | Load and stress testing                 |
| 💻 **Open WebUI**        | Frontend integration                    |

---

## 💡 Hardware Requirements

Depending on which part of the RAG server you're running, the hardware demands vary. Here’s a breakdown of what benefits from GPU acceleration and what runs fine on CPU:

| Component                          | Requires GPU? | Notes                                                                 |
|------------------------------------|---------------|-----------------------------------------------------------------------|
| **Embedding Model Inference** (`sentence-transformers/all-mpnet-base-v2`) | Recommended   | Runs on CPU, but GPU is strongly recommended for faster document embedding, especially for large datasets. |
| **Vector Search (FAISS)**        | No            | FAISS typically runs on CPU; GPU FAISS is optional for large-scale production systems. |
| **LLM Response Generation** (`TechxGenus_Mistral-Large-Instruct-2407-AWQ`) | Recommended   | Model inference is significantly faster on GPU. Small setups may run on CPU but expect higher latency. |
| **API Server (FastAPI)**         | No            | The server logic is CPU-based and doesn’t require GPU. |
| **Cache Layer (Redis or Similar)** | No            | Runs entirely on CPU, designed for low latency key-value storage. |

⚠️ **Note:** While the RAG server can run entirely on CPU, for production or large document sets, a GPU is highly recommended to reduce inference and retrieval latency.

---

## 🧑‍💻 Installation Guide

1️⃣ Clone the repository:

```bash
git clone https://github.com/LisaRebecca/rag-server.git
cd rag-server
```


2️⃣ For **local** setup please follow the following steps:

⚠️ **Note:** Make sure you have Python installed (recommendation: Python 3.9+)

a. Inside your project's **root** directory create a **Virtual Environment**:

```bash
python3 -m venv myvenv
```

b. Activate the Virtual Environment:

i. On Linux/**macOS**:

`source myvenv/bin/activate`

ii. On **Windows**:

`myvenv\Scripts\activate`

c. From your project's **root** directory install all required packages:

```bash
pip install -r requirements.txt
```

d. Install Open WebUI:

```bash
pip install open-webui
```

e. Start the Open WebUI server:

```bash
open-webui serve
```


3️⃣ Run the application locally:

Once the dependencies are installed, start the server using:

```bash
uvicorn main:app --reload
```


4️⃣ Finally, open your browser and navigate to: 
`http://localhost:8090`

The API documentation and endpoints will be available at: 
`http://localhost:8090/docs`

---

## 📚 Knowledge Base Setup

Download the **chunked JSONL knowledge base** here:

👉 [Download Knowledge Base](https://drive.google.com/file/d/1_4BNVhkEaAOngTAsgLgh38kWe0aQdrqW/view?usp=drive_link)

Inside `src/fau_rag` create a new folder labeled `knowledgebase` and drop the newly downloaded knowledge base inside of it.

If `src/fau_rag/knowledgebase` already exists just drop the knowledgebase file inside of it.

---

## 🔍 FAISS Indices & Models

Select your preferred index depending on speed vs. embedding quality:

| Index Type         | Dimensions | Model                     | Download Link                                               |
|---------------------|------------|----------------------------|-------------------------------------------------------------|
| **Light Index**   | 384        | `all-MiniLM-L6-v2`         | [Download](https://drive.google.com/file/d/1qOECFQ_Df_sBCextiqRbPjeKHTFXpbdW/view?usp=sharing) |
| **Normal Index**  | 768        | `all-mpnet-base-v2`        | [Download](https://drive.google.com/file/d/1-0ncb5rZ-9SSosAocHnuR6iYIfLLdtNE/view?usp=sharing) |

After successfully downloading the FAISS file, drop it inside `src/fau_rag/knowledgebase`

---

## 🌐 Connecting to FAU Endpoints (via Cisco VPN)

1️⃣ **Install Cisco AnyConnect Client**

Connect to:

```bash
vpn.fau.de
```


2️⃣ **Login with FAU Credentials**

Use your FAU `IdM` username and password.

---

## 🐳 Running the Service with Docker

### 🔨 Build Docker Images (First Time Only)

From your project directory: `cd src/fau_rag`

1. **Embedding Service:**
```bash
docker build -f embedding_service/Dockerfile -t embedding-service .
```
To run/test the individual service, from `src/fau_rag/embedding_service` run:
```bash
docker-compose up -d
```

2. **RAG Service:**
```bash
docker build -f rag_service/Dockerfile -t rag-service .
```

To run/test the individual service, from `src/fau_rag/rag_service` run:
```bash
docker-compose up -d
```

### 🚀 Run the Full Service with Docker Compose (For Existing Builds)

From your project's **root** directory:

```bash
docker-compose up -d
```

---

## 🖼️ Open WebUI Setup (Frontend for OpenAI API)

1. Run this container for the Open WebUI interface:

```bash
docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key \
-v open-webui:/app/backend/data \
--name open-webui --restart always ghcr.io/open-webui/open-webui:main
```


2. Navigate to:
`http://localhost:3000`


3. Sign in with your Name, Email, and Password


4. To connect the services together, on the bottom left corner of the Open WebUI frontend:

    a. Goto: Admin Panel -> Settings -> Connections -> Manage OpenAI API Connections

    b. Add new connection:

    URL: `http://rag-service:8090/v1`

    KEY: "YOUR API KEY HERE" must match the **OPENAI_API_KEY** you're using inside your **.env** file.

After verifying the connection you should be able to find "**FAU LLM 2.0**" in the dropdown list of models.

---

## 💬 Support & Contributions

We welcome contributions, feedback, and collaborations!  
For issues or questions, please open an [Issue](https://github.com/LisaRebecca/rag-server/issues) or submit a Pull Request.

---
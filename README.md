# 🚀 Welcome to **Tier0** — FAU's RAG LLM System

Tier0 is a **modular Retrieval-Augmented Generation (RAG) system** crafted to optimize the trade-off between **retrieval speed** and **response accuracy**. Designed with flexibility and scalability in mind, this system empowers AI-driven information querying — perfect for both academic and enterprise-grade environments.

---

## 🧠 Key Features

- ⚡ **Fast & Accurate** — balances retrieval speed and relevance for optimal AI responses.
- 🧩 **Modular Architecture** — easy to extend, swap, and maintain components.
- 🔐 **Enterprise-Ready** — integrates smoothly into secure networks like FAU's infrastructure.
- 🌍 **Flexible Frontend** — Open WebUI compatibility for user-friendly interaction.

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

## 🧑‍💻 Installation Guide

1️⃣ Clone the repository:

```bash
git clone https://github.com/LisaRebecca/rag-server.git
cd rag-server
```

---

## 📚 Knowledge Base Setup

Download the **chunked JSONL knowledge base** here:

👉 [Download Knowledge Base](https://drive.google.com/file/d/1_4BNVhkEaAOngTAsgLgh38kWe0aQdrqW/view?usp=drive_link)

---

## 🔍 FAISS Indices & Models

Select your preferred index depending on speed vs. embedding quality:

| Index Type         | Dimensions | Model                     | Download Link                                               |
|---------------------|------------|----------------------------|-------------------------------------------------------------|
| 🌱 **Light Index**   | 384        | `all-MiniLM-L6-v2`         | [Download](https://drive.google.com/file/d/1qOECFQ_Df_sBCextiqRbPjeKHTFXpbdW/view?usp=sharing) |
| 🌳 **Normal Index**  | 768        | `all-mpnet-base-v2`        | [Download](https://drive.google.com/file/d/1-0ncb5rZ-9SSosAocHnuR6iYIfLLdtNE/view?usp=sharing) |

➡️ After downloading, update your path in:

```python
VECTORSTORE_PATH
```
Located in `server/fastapi_router.py`.

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

From your project directory:

1. **Embedding Service:**
```bash
docker build -f embedding_service/Dockerfile -t embedding-service .
```

2. **FastAPI RAG App:**
```bash
docker build -f fastapi_RAG_service/Dockerfile -t fastapi-rag-app .
```

### 🚀 Run with Docker Compose (For Existing Builds)

```bash
docker-compose up -d
```

---

## 🖼️ Open WebUI Setup (Frontend for OpenAI API)

Run this container for the Open WebUI interface:

```bash
docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key \
-v open-webui:/app/backend/data \
--name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

---

## 💡 How to Use

Once the RAG system is up and running:

1. Open your browser.
2. Navigate to:
```
http://localhost:3000
```
3. Start querying your knowledge base via the Open WebUI frontend!

---

## 💬 Support & Contributions

We welcome contributions, feedback, and collaborations!  
For issues or questions, please open an [Issue](https://github.com/LisaRebecca/rag-server/issues) or submit a Pull Request.

---
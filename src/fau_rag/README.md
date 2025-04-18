# 🧩 Internal Service Overview — Tier0 RAG System

This document outlines the **four core services** that power our modular Retrieval-Augmented Generation (RAG) system.  
Each service is responsible for a specific task, making the system easy to scale, maintain, and optimize.

---

## 💻 1️⃣ Open WebUI — *User Interaction Layer - Frontend*

**Purpose:**  
Acts as the entry point for users to interact with the FAU LLM RAG system.

**Responsibilities:**

- Receives plain text queries from the user via a web interface.
- Forwards user queries to the backend services for processing.
- Displays final responses from the RAG Service to the user.

**Tech Highlights:**

- Web-based, OpenAI-compatible chat interface.

---

## 📐 2️⃣ Embedding Service — *Vector Conversion*

**Purpose:**  
Transforms raw text queries into high-dimensional embedding vectors suitable for semantic search.

**Responsibilities:**

- Receives plain text queries from Open WebUI.
- Uses `sentence-transformers/all-mpnet-base-v2` to generate embedding vectors.
- Returns the embedding vector back to the Retrieval Service.

**Tech Highlights:**

- Powered by `sentence-transformers` for dense embeddings.
- GPU-acceleration recommended for optimal performance.
- Offers a RESTful API endpoint for easy integration.

**Interaction [Input/Output]:**
```bash
POST /embed
{
  "query": "Tell me about optimization courses at FAU."
}
=> returns
{
  "embedding": [0.023, -0.124, ...]
}
```

---

## 🔎 3️⃣ Retrieval Service — *Similarity Search Engine*

**Purpose:**  
Performs fast and accurate document retrieval based on embedding similarity.

**Responsibilities:**

- Receives embedding vectors from the Embedding Service.
- Performs similarity search using **FAISS** to locate relevant documents.
- Returns the top matching documents to the RAG Service for answer generation.

**Tech Highlights:**

- Utilizes **FAISS** (Facebook AI Similarity Search) for vector-based retrieval.
- Supports configurable similarity metrics and index types.

**Interaction [Input/Output]:**
```bash
POST /retrieve
{
  "embedding": [0.023, -0.124, ...]
}
=> returns
{
  "documents": [
    {"content": "FAU offers various optimization courses...", "metadata": {...}},
    ...
  ]
}
```

---

## 🧠 4️⃣ RAG Service — *Response Generation Engine*

**Purpose:**  
Generates responses based on retrieved documents and the original query.

**Responsibilities:**

- Receives both the original query and relevant documents from the Retrieval Service.
- Uses the LLM `TechxGenus_Mistral-Large-Instruct-2407-AWQ` to formulate an answer.
- Returns the final response back to Open WebUI in `ChatCompletionResponse` format.

**Tech Highlights:**

- Modular design allows swapping LLMs without altering system flow.

**Interaction [Input/Output]:**
```bash
POST /generate
{
  "query": "Tell me about optimization courses at FAU.",
  "documents": [
    {"content": "FAU offers various optimization courses...", "metadata": {...}},
    ...
  ]
}
=> returns
{
  "response": "FAU's optimization courses include..."
}
```

---

## 🕸️ System Flow Overview

User Input (Open WebUI)
          ⬇️
Embedding Vector (Embedding Service)
          ⬇️
Relevant Documents (Retrieval Service)
          ⬇️
Final Answer (RAG Service)
          ⬇️
Displayed to User (Open WebUI)


---

## 📢 Notes

- Each service is containerized and deployable independently.
- Each service can be tested independently on it's own.
- Designed for asynchronous, parallelizable request handling.
- Modular architecture allows easy replacement, scaling, and monitoring.

---

✅ **Tip:**  
For optimal performance in production, ensure GPU resources are allocated to the **Embedding Service** and **RAG Service**.

---
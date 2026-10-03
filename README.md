# 🤖 Nexora

**Nexora** is an open-source **agentic AI assistant** built with **Python, FastAPI, LangGraph, LangChain, Google Gemini, Tavily, ChromaDB, SQLite, and LangSmith**.

It combines conversational AI, real-time streaming, document-based question answering, web search, RAG, conversation memory, and application tracing into a single AI assistant.

---

## ✨ Features

* 💬 **AI Conversation** — Chat with an AI agent powered by Google Gemini
* ⚡ **Real-Time Streaming** — Stream AI responses as they are generated
* 📄 **Document Upload** — Upload PDF, DOCX, TXT, MD, PY, and CSV files
* 🔎 **RAG** — Ask questions based on uploaded documents
* 🌐 **Web Search** — Search the web using Tavily
* 🧠 **Conversation Memory** — Store and retrieve conversation history
* 🤖 **Agentic Workflow** — LangGraph-based agent orchestration
* 📚 **Vector Search** — ChromaDB for document retrieval
* 🗄️ **Persistence** — SQLite for conversation data
* 📊 **LLM Observability** — LangSmith for tracing and monitoring
* 🎨 **Web Interface** — Jinja2-based frontend
* 🚀 **FastAPI Backend**
* 🐳 **Docker Ready**

---

# 🛠️ Tech Stack

| Technology        | Purpose                                  |
| ----------------- | ---------------------------------------- |
| **Python**        | Core application development             |
| **FastAPI**       | Backend server and API endpoints         |
| **Jinja2**        | Web interface rendering                  |
| **LangGraph**     | Agent orchestration                      |
| **LangChain**     | LLM integration, tools and RAG           |
| **Google Gemini** | AI / LLM provider                        |
| **Tavily**        | Web search                               |
| **ChromaDB**      | Vector database and document retrieval   |
| **SQLite**        | Conversation persistence                 |
| **LangSmith**     | LLM tracing, debugging and observability |
| **Docker**        | Containerization                         |

---

# 🧠 Architecture

```text
                         ┌──────────────────┐
                         │       User       │
                         │    Web Browser   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │     Backend      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    LangGraph     │
                         │      Agent       │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
             ┌───────────┐ ┌───────────┐ ┌───────────┐
             │  Gemini   │ │  Tavily   │ │    RAG    │
             │    LLM    │ │ Web Search│ │  Pipeline  │
             └───────────┘ └───────────┘ └─────┬─────┘
                                               │
                                               ▼
                                         ┌───────────┐
                                         │ ChromaDB  │
                                         └───────────┘

                         ┌──────────────────┐
                         │      SQLite      │
                         │ Conversation DB  │
                         └──────────────────┘

                         ┌──────────────────┐
                         │    LangSmith     │
                         │ Tracing & Debug  │
                         └──────────────────┘
```

---

# 📊 LangSmith Observability

Nexora integrates **LangSmith** to provide visibility into the AI application's execution.

LangSmith can be used to:

* 🔍 Trace LangChain and LangGraph executions
* 🧩 Debug agent workflows
* 📈 Monitor LLM calls
* ⏱️ Inspect execution latency
* 📝 Review inputs and outputs
* 🛠️ Identify issues in agent/tool execution

This makes it easier to understand and debug the behavior of the agent during development.

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=your_gemini_model

TAVILY_API_KEY=your_tavily_api_key

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=nexora
```

### LangSmith Configuration

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=nexora
```

Replace:

```text
your_langsmith_api_key
```

with your actual LangSmith API key.

> ⚠️ Never commit your `.env` file or API keys to GitHub.

---

# 📂 Project Structure

```text
Nexora/
│
├── app.py                  # FastAPI application
├── agent.py                # LangGraph agent and orchestration
├── database.py             # SQLite database and persistence
├── rag.py                  # Document processing and RAG
├── tools.py                # Agent tools and web search
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── .dockerignore           # Docker ignore rules
├── .env                    # Environment variables
│
├── templates/
│   └── index.html          # Web interface
│
├── uploads/                # Uploaded documents
├── data/                   # SQLite/application data
└── chroma_db/              # ChromaDB vector storage
```

---

# 🚀 Getting Started

## Prerequisites

Install:

* Python
* pip or Conda
* Git
* Google Gemini API key
* Tavily API key
* LangSmith API key

Docker is optional.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Sahil489s/Nexora.git
cd Nexora
```

---

## 2. Create a Virtual Environment

### Conda

```bash
conda create -n nexora python=3.11 -y
conda activate nexora
```

### Python venv

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8080
```

---

# 📄 RAG Pipeline

Nexora uses Retrieval-Augmented Generation to answer questions using uploaded documents.

```text
Upload Document
       │
       ▼
Document Loading
       │
       ▼
Text Extraction
       │
       ▼
Text Chunking
       │
       ▼
Embeddings
       │
       ▼
ChromaDB
       │
       ▼
Similarity Search
       │
       ▼
Relevant Context
       │
       ▼
Gemini
       │
       ▼
AI Response
```

---

# 🌐 Web Search

Nexora uses Tavily for web search.

Example:

```text
Search the web for the latest AI agent developments.
```

The retrieved information can be passed to the agent as context before generating the final response.

---

# 🧠 Conversation Memory

Nexora uses SQLite to persist conversation data.

```text
User
 │
 ▼
LangGraph Agent
 │
 ▼
Gemini / Tools
 │
 ▼
Response
 │
 ▼
SQLite
 │
 ▼
Conversation History
```

---

# 🐳 Docker

Build the image:

```bash
docker build -t nexora .
```

Run the container:

```bash
docker run -d \
  --name nexora \
  --restart always \
  -p 8080:8080 \
  --env-file .env \
  nexora
```

Open:

```text
http://localhost:8080
```

---

# 💡 Example Questions

### General AI

```text
Explain machine learning in simple terms.
```

### Document Analysis

```text
Summarize the uploaded PDF.
```

### Document Q&A

```text
Based on my uploaded document, what are the key findings?
```

### Web Search

```text
Search the web for the latest developments in generative AI.
```

### Calculation

```text
Calculate 125 * 48 / 6.
```

---

# 🔐 Security

* Never commit `.env` to GitHub.
* Keep Gemini, Tavily, and LangSmith API keys private.
* Never expose API keys in frontend code.
* Rotate keys if accidentally exposed.
* Use proper secret management for production deployments.

Recommended `.gitignore`:

```gitignore
.env

__pycache__/
*.pyc

venv/
.venv/

chroma_db/
data/
uploads/
```

---

# 🚧 Project Status

## Implemented

* ✅ Google Gemini integration
* ✅ LangGraph agent
* ✅ LangChain
* ✅ LangSmith tracing and observability
* ✅ Real-time streaming
* ✅ Document upload
* ✅ RAG-based document Q&A
* ✅ ChromaDB vector storage
* ✅ Tavily web search
* ✅ SQLite conversation persistence
* ✅ FastAPI backend
* ✅ Jinja2 frontend
* ✅ Docker configuration

## Future Improvements

* 🔐 User authentication
* 👥 Multi-user support
* ☁️ Cloud deployment
* 🔄 CI/CD pipeline
* 📊 Advanced monitoring
* 📈 Production scaling
* 🗂️ Improved document management

---

# 🤝 Contributing

Contributions are welcome.

```bash
git checkout -b feature/your-feature
git add .
git commit -m "Add new feature"
git push origin feature/your-feature
```

Then open a Pull Request.

---

# 👨‍💻 Author

**Sahil Sharma**

GitHub:
https://github.com/Sahil489s

LinkedIn:
https://www.linkedin.com/in/sahil489/

---

# ⭐ Support

If you find **Nexora** useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is open source. Please check the repository license for usage terms.
